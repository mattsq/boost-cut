"""The Dust track-1 record (train/val/run.py, unchanged: its estimator, optimizer, population and schedule) at any
preset of the Dust paper's ladder and any seed, scored like train/boostcut/run.py (validation and a held-out test
split). The record's per-block draw lists are repeated for the preset's depth.

    torchrun --standalone --nproc_per_node=8 analysis/dust_preset.py --preset 243m --output /data/runs/dust243
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('record', ROOT / 'train/val/run.py')
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from utils.model import GPT, GPTConfig  # noqa: E402
from utils.prepare_data import SEQ, TRAIN_ROWS, VAL_ROWS  # noqa: E402


def make_model(device, seed, config, cls=GPT):
    """The record's make_model for any config: its init, with uniform output projections."""
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = cls(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=sorted(bc.PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0, help='Population: the 16k cell times this.')
    parser.add_argument('--test-file', default='')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many updates.')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--arch', choices=('noise', 'plain'), default='noise', help='plain: analysis/plain_gpt.py.')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    n_layer, width = bc.PRESETS[args.preset]
    config = GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width)
    cls = GPT
    if args.arch == 'plain':
        spec_p = importlib.util.spec_from_file_location('plain_gpt', ROOT / 'analysis/plain_gpt.py')
        plain = sys.modules['plain_gpt'] = importlib.util.module_from_spec(spec_p)
        spec_p.loader.exec_module(plain)
        plain.patch_dust(rec)
        cls = plain.PlainGPT
        config = GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width, window_pattern='L')
    settings = rec.recipe(args.scale)
    settings.writers, settings.hidden, settings.hubs = (lst[:1] * n_layer for lst in
                                                        (settings.writers, settings.hidden, settings.hubs))
    training = bc.load_split(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(rec.SEED + 1))]
    val_rows = bc.load_split(args.data_dir / 'val.pt', VAL_ROWS)
    steps = min(len(training) // rec.BATCH, args.steps or len(training))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            method='dust_record', arch=args.arch, preset=args.preset, model=asdict(config), seed=args.seed,
            batch=rec.BATCH, steps=steps, world=world, recipe=asdict(settings)), indent=2) + '\n')

    def init():
        model = make_model(device, args.seed, config, cls)
        return model, rec.make_optimizer(model, settings, steps), rec.DUST(model, settings, args.seed, rank, world)

    def set_rate(optimizer, step):
        left = (steps - step + 1) / steps
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / settings.warmdown if settings.warmdown else 1.0)

    model, optimizer, estimator = init()
    noise = torch.randint(0, config.vocab_size, (rec.BATCH, SEQ), generator=torch.Generator().manual_seed(rec.SEED)).to(device)
    estimator.step(noise[:, :-1].contiguous(), noise[:, 1:].contiguous(), optimizer)
    estimator.close()
    model, optimizer, estimator = init()
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        batch = training[(step - 1) * rec.BATCH:step * rec.BATCH].to(device)
        set_rate(optimizer, step)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and step % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record['val_loss'] = bc.evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
    estimator.close()
    result = dict(val_loss=bc.evaluate(model, val_rows), train_seconds=round(elapsed, 2), steps=steps)
    if args.test_file:
        result['test_loss'] = bc.evaluate(model, bc.load_split(args.data_dir / args.test_file))
    if rank == 0:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'final.pt')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
