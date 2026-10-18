"""Peak GPU memory per training step for each method (analysis only): one sequence of 2,048 tokens on one GPU, each
method's work on one sequence (backprop: forward and backward; Boost: its full cycle on one GPU, where training
splits it over two; Cut: as in training), after a warmup call, measured by torch's allocator (weights included,
optimizer state excluded).

    python analysis/memory.py --preset 38m
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from utils.model import GPTConfig  # noqa: E402
from utils.prepare_data import TRAIN_ROWS  # noqa: E402


def peak(fn):
    """Peak allocated memory (GiB) during a second call of fn, after a warmup call."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--preset', default='38m')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0)
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    config = GPTConfig(n_layer=n_layer, n_head=bc.WIDTH // 64, n_kv_head=bc.WIDTH // 64, n_embd=bc.WIDTH)
    rows = bc.load_split(args.data_dir / 'train.pt', TRAIN_ROWS)[:1].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=bc.HIDDEN // 16)
    out = dict(preset=args.preset)
    model = bc.make_model(device, 0, config)

    def backprop():
        model.requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(x, y).backward()
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    out['weights_gib'] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    for name in ('backprop', 'boost', 'cut'):
        est = None if name == 'backprop' else (bc.Boost if name == 'boost' else bc.Cut)(model, settings, 0, 0, 1)
        out[name] = dict(peak_gib=round(peak(backprop if est is None else lambda: est.estimate(x, y)), 2))
        print(json.dumps({name: out[name]}), flush=True)
        if est is not None:
            est.close()
        del est
        torch.cuda.empty_cache()
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
