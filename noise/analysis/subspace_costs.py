"""Cost of subspace credit against the full cycle (analysis only; docs/proposals/circuit-guided-low-rank-credit.md,
sections 5.2 and 8.2): for Boost and Cut at each probe rank, per training token,
- counted matmul FLOPs of one estimate (analysis/flops.py's counter: eager, dense attention), and vocabulary entries
  evaluated (the soft-capped cross-entropies, counted from the probe shapes);
- the bank's own work: counted FLOPs of gathering one step's statistics, and the time of a refresh (eigh is not a
  matmul, so it is timed, not counted);
- with --time (GPU), wall time of one compiled estimate after a warmup estimate at that shape (the compilation's own
  time reported separately), with the estimator's time split.
Ranks change the number of probe directions only, so a random basis stands for every kind (oracle-grad is the
exception: it measures the full cycle).

    python analysis/subspace_costs.py --preset 38m --ranks 16,32,64,128 --time
    python analysis/subspace_costs.py --preset 2m --ranks 16,32 --random-tokens --seq-len 128   # smoke test (CPU)
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import torch._dynamo
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train/boostcut'))
spec = importlib.util.spec_from_file_location('flops', ROOT / 'analysis/flops.py')
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)
bc = fl.bc
from projected_credit import make_estimators  # noqa: E402
from subspace import SubspaceBank  # noqa: E402
import utils.model as um  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def counted(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    sync()
    return counter.get_total_flops()


def timed(fn):
    sync()
    start = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--preset', default='38m')
    parser.add_argument('--ranks', default='16,32,64,128')
    parser.add_argument('--kinds', default='route-cov,activation-pca', help='Bases whose statistics are costed.')
    parser.add_argument('--time', action='store_true', help='Also time compiled estimates (GPU).')
    parser.add_argument('--seq-len', type=int, default=0)
    parser.add_argument('--random-tokens', action='store_true')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    L, W = n_layer, bc.WIDTH
    config = bc.GPTConfig(n_layer=n_layer, n_head=W // 64, n_kv_head=W // 64, n_embd=W)
    V = config.vocab_size
    if args.random_tokens:
        rows = torch.randint(0, V, (1, (args.seq_len or 2048) + 1), generator=torch.Generator().manual_seed(0))
    else:
        rows = bc.load_split(args.data_dir / 'train.pt')[:1]
        rows = rows[:, :args.seq_len + 1] if args.seq_len else rows
    rows = rows.to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    tokens = x.numel()
    settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=bc.HIDDEN // 16, cut_exact_keys=True)
    Boost, Cut = make_estimators(bc)
    ranks = [r for r in map(int, args.ranks.split(',')) if r < W] + [W]

    def estimator(phase, rank):
        if rank == W:
            bank = SubspaceBank('none', W, L, 1, device=device)
        else:
            bank = SubspaceBank('random', W, L, rank, device=device)
            bank.refresh()
        model = bc.make_model(device, 0, config)
        return Boost(model, settings, 0, 0, 1, bank=bank, enabled=True) if phase == 'boost' else \
            Cut(model, settings, 0, 0, 1, bank=bank, enabled=True)

    def vocab_entries(phase, P):
        if phase == 'boost':  # readouts at mid and out probed at P, hubs at the full W; coordinate heads; the head
            return (2 * L * P + L * W) * V + (2 * L - 1) * V + V
        return (L + 1) * (P + 4) * V + P * V + V  # chain sites with the scalar moves, the top stream, the head

    results = []
    # counted FLOPs (eager, dense attention)
    saved = {m: {k: getattr(m, k) for k in ('attention', 'flex', 'window_mask') if hasattr(m, k)} for m in (um, bc)}
    torch._dynamo.config.disable = True
    fl.patch(um, bc)
    try:
        for phase in ('boost', 'cut'):
            for r in ranks:
                est = estimator(phase, r)
                flops = counted(lambda: est.estimate(x, y)) / tokens
                est.close()
                row = dict(phase=phase, rank=r, flops_per_token=round(flops), vocab_evals_per_token=vocab_entries(phase, r))
                results.append(row)
                print(json.dumps(row), flush=True)
        model = bc.make_model(device, 0, config)
        est = Boost(model, settings, 0, 0, 1, bank=SubspaceBank('none', W, L, 1, device=device), enabled=False)
        est.estimate(x, y)
        for kind in args.kinds.split(','):
            bank = SubspaceBank(kind, W, L, max(ranks[:-1] or [W]), warmup=1, route_k=settings.route_k, device=device)
            est.bank = bank
            flops = counted(lambda: est.observe(y)) / tokens
            seconds = timed(bank.refresh)
            row = dict(statistics=kind, flops_per_token=round(flops), refresh_seconds=round(seconds, 4))
            results.append(row)
            print(json.dumps(row), flush=True)
        est.close()
    finally:
        for m, attrs in saved.items():
            for k, v in attrs.items():
                setattr(m, k, v)
        torch._dynamo.config.disable = False
    # wall time of compiled estimates
    if args.time:
        if not cuda:
            raise SystemExit('--time needs a GPU (compiled kernels).')
        for phase in ('boost', 'cut'):
            for r in ranks:
                est = estimator(phase, r)
                compile_seconds = timed(lambda: est.estimate(x, y))
                est.profile, est.times = True, {}
                seconds = timed(lambda: est.estimate(x, y))
                est.profile = False
                row = dict(phase=phase, rank=r, seconds=round(seconds, 4), first_call_seconds=round(compile_seconds, 2),
                           parts={k: round(v, 4) for k, v in est.times.items()})
                results.append(row)
                print(json.dumps(row), flush=True)
                est.close()
    print(json.dumps(dict(preset=args.preset, tokens=tokens, results=results)), flush=True)


if __name__ == '__main__':
    main()
