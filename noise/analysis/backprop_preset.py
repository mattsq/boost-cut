"""Backprop baseline at any preset (analysis only, never a Noise entry): the same model, init, optimizer (ANVIL II plus
Adam, the record's rates and schedule), data order and scoring as train/boostcut/run.py, with the true gradient.
Data parallel over the GPUs (each takes its share of the batch's sequences). Timed like the forward-only runs.

    torchrun --standalone --nproc_per_node=8 analysis/backprop_preset.py --preset 243m --output /data/runs/bp243
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from utils.prepare_data import SEQ, TRAIN_ROWS, VAL_ROWS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=sorted(bc.PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=bc.SEED)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--polar-margin', type=float, default=bc.POLAR_MARGIN)
    parser.add_argument('--test-file', default='')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many sequences.')
    parser.add_argument('--lr-mult', type=float, default=1.0, help='Scale both learning rates (Adam and ANVIL II).')
    parser.add_argument('--arch', choices=('noise', 'plain'), default='noise', help='plain: analysis/plain_gpt.py.')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    bc.POLAR_MARGIN = args.polar_margin
    if args.arch == 'plain':
        spec_p = importlib.util.spec_from_file_location('plain_gpt', ROOT / 'analysis/plain_gpt.py')
        plain = sys.modules['plain_gpt'] = importlib.util.module_from_spec(spec_p)
        spec_p.loader.exec_module(plain)
        plain.patch(bc)
    n_layer, width = bc.PRESETS[args.preset]
    config = bc.GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width)
    recipe = bc.Recipe()
    recipe.lr, recipe.anvil_lr = recipe.lr * args.lr_mult, recipe.anvil_lr * args.lr_mult
    training = bc.load_split(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(bc.SEED + 1))]
    val_rows = bc.load_split(args.data_dir / 'val.pt', VAL_ROWS)
    steps = len(training) // args.batch
    per = args.batch // world
    model = bc.make_model(device, args.seed, config)
    optimizer = bc.make_optimizer(model, recipe, steps)
    params = list(model.named_parameters())
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            method='backprop', arch=args.arch, preset=args.preset, model=asdict(config), seed=args.seed,
            batch=args.batch, steps=steps, world=world, recipe=asdict(recipe), polar_margin=bc.POLAR_MARGIN), indent=2)
            + '\n')
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None

    def step_fn(batch):
        model.requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        loss.backward()
        model.requires_grad_(False)
        flat = torch.cat([(p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)).reshape(-1)
                          for _, p in params])
        if world > 1:
            torch.distributed.all_reduce(flat)
            flat /= world
        offset = 0
        for _, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return bc.average_ranks(loss.detach().float().reshape(1)).item()

    # untimed warmup on random tokens (compilation), then a fresh model
    noise = torch.randint(0, config.vocab_size, (per, SEQ), generator=torch.Generator().manual_seed(bc.SEED)).to(device)
    step_fn(noise)
    model = bc.make_model(device, args.seed, config)
    optimizer = bc.make_optimizer(model, recipe, steps)
    params = list(model.named_parameters())
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        left = (steps - step + 1) / steps
        for g in optimizer.param_groups:
            g.setdefault('initial_lr', g['lr'])
            g['lr'] = g['initial_lr'] * min(1.0, left / recipe.warmdown)
        first = (step - 1) * args.batch + rank * per
        loss = step_fn(training[first:first + per].to(device))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and (step * args.batch) % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record['val_loss'] = bc.evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        if rank == 0:
            log.write(json.dumps(record) + '\n')
            log.flush()
            if step % 61 == 0 or 'val_loss' in record:
                print(json.dumps(record), flush=True)
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
