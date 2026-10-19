"""Circuit-guided subspace credit (CGSC): Boost -> Cut with residual-stream credit measured in per-site subspaces, and
optionally rank-restricted writer updates (docs/proposals/circuit-guided-low-rank-credit.md). Experimental entry:
train/boostcut/run.py and entries/ are unchanged, and everything not named here is theirs (model, init, data order,
schedule, handoff, evaluation).

    torchrun --standalone --nproc_per_node=8 train/cgsc/run.py --credit-basis route-cov --credit-rank 64 \\
        --basis-refresh-steps 128 --output logs/cgsc_route64
    torchrun --standalone --nproc_per_node=8 train/cgsc/run.py --credit-basis route-cov --credit-rank 64 \\
        --update-mode qa-merge --adapter-merge-steps 128 --output logs/cgsc_qa64

With --credit-basis full and --update-mode dense (the defaults) the run is train/boostcut/run.py's: every probe takes
the original code path. --credit-basis oracle-grad measures the full cycle and projects it (an upper bound, never a
forward-only result); its config and result are marked diagnostic_only. Time spent on basis statistics, refreshes and
exploration probes is training time; the leakage and effective-rank diagnostics are not (logged as diag_seconds).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train/boostcut'))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = sys.modules['boostcut'] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from adapters import MODES, UpdateLeakage, WriterUpdates, writer_names  # noqa: E402
from projected_credit import make_estimators  # noqa: E402
from subspace import EXPLORE, SubspaceBank  # noqa: E402
from utils.prepare_data import SEQ, TRAIN_ROWS, VAL_ROWS  # noqa: E402

BASES = ('full', 'random', 'activation-pca', 'contrib', 'route-cov', 'oracle-grad')


def make_optimizer(model, recipe, steps, mode, bank, sites, args):
    """The record's optimizer (ANVIL II on block matrices, Adam elsewhere); under a restricted update mode the
    writers leave ANVIL for WriterUpdates (which steps them with their own ANVIL until the adapters engage)."""
    restricted = set(writer_names(model.config.n_layer)) if mode != 'dense' else set()
    matrices, writers, groups = [], [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            (writers if name in restricted else matrices).append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW([{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
                             betas=recipe.betas, eps=1e-8, weight_decay=0.0)
    parts = [bc.Anvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps), adam]
    if restricted:
        fallback = bc.Anvil(writers, recipe.anvil_lr, recipe.anvil_wd, steps)
        parts.append(WriterUpdates(model, bank, mode, args.adapter_rank, sites, args.adapter_lr, recipe.betas,
                                   args.adapter_merge_steps, fallback, seed=args.seed))
    optimizer = bc.Optimizers(*parts)
    if args.leakage_every:
        optimizer = UpdateLeakage(optimizer, model, bank, sites, args.leakage_every)
    return optimizer


def parts_of(optimizer):
    """(inner Optimizers, WriterUpdates or None, UpdateLeakage or None)."""
    leakage = optimizer if isinstance(optimizer, UpdateLeakage) else None
    inner = optimizer.optimizer if leakage else optimizer
    writers = next((o for o in inner.optimizers if isinstance(o, WriterUpdates)), None)
    return inner, writers, leakage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/cgsc'))
    parser.add_argument('--preset', choices=sorted(bc.PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=bc.SEED)
    parser.add_argument('--switch', type=float, default=0.3, help='Fraction of the epoch trained by Boost.')
    parser.add_argument('--route-k', type=int, default=0, help='Cut: MLP units per token per block (0: hidden/16).')
    parser.add_argument('--keys', choices=('exact', 'sampled'), default='exact', help='Cut\'s keys.')
    parser.add_argument('--boost-keys', choices=('exact', 'sampled'), default='exact', help='Boost\'s keys.')
    parser.add_argument('--train-file', default='train.pt')
    parser.add_argument('--train-rows', type=int, default=TRAIN_ROWS)
    parser.add_argument('--test-file', default='', help='Also score the final weights on this split (e.g. test.pt).')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many sequences.')
    parser.add_argument('--chain-tokens', type=int, default=0)
    parser.add_argument('--key-chunk', type=int, default=0)
    parser.add_argument('--steps', type=int, default=0, help='Smoke test: stop after this many updates.')
    parser.add_argument('--save-every', type=int, default=0, help='Save the weights and the subspace state every '
                        'this many sequences.')
    parser.add_argument('--lr-mult', type=float, default=1.0, help='Scale both learning rates.')
    # circuit-guided subspace credit
    parser.add_argument('--credit-basis', choices=BASES, default='full', help='Probe basis at residual-stream sites '
                        '(full: the original full cycle; oracle-grad: diagnostic upper bound only).')
    parser.add_argument('--credit-rank', type=int, default=64, help='Probe directions per active site.')
    parser.add_argument('--basis-refresh-steps', type=int, default=0, help='Rebuild the bases every this many '
                        'updates after the first (0: never).')
    parser.add_argument('--basis-warmup-steps', type=int, default=16, help='Full-cycle updates before the first '
                        'basis (statistics gather meanwhile).')
    parser.add_argument('--exploration-probe-fraction', type=float, default=0.0, help='Share of each active site\'s '
                        'probes moved along random directions outside its basis.')
    parser.add_argument('--exploration-use', choices=EXPLORE, default='measure')
    parser.add_argument('--subspace-target', choices=('boost', 'cut', 'both'), default='both')
    parser.add_argument('--update-mode', choices=MODES, default='dense')
    parser.add_argument('--adapter-rank', type=int, default=0, help='Writer update rank (0: the credit rank).')
    parser.add_argument('--adapter-merge-steps', type=int, default=0, help='qa-merge / lora: merge every this many '
                        'updates (the handoff always merges).')
    parser.add_argument('--adapter-lr', type=float, default=0.0, help='AdamW (or SGD) rate of the writer updates '
                        '(0: the record\'s Adam rate).')
    parser.add_argument('--leakage-every', type=int, default=0, help='Measure the writers\' update leakage out of '
                        'their credit subspace every this many updates (untimed).')
    parser.add_argument('--profile-every', type=int, default=0, help='Log a synchronized time split every this many '
                        'updates.')
    args = parser.parse_args()

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = torch.cuda.is_available()  # without a GPU: a CPU smoke test (eager, slow; not a timing)
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    sync = torch.cuda.synchronize if cuda else lambda: None
    if cuda:
        torch.cuda.set_device(device)
    else:
        torch._dynamo.config.disable = True
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    WIDTH = bc.WIDTH
    config = bc.GPTConfig(n_layer=n_layer, n_head=WIDTH // 64, n_kv_head=WIDTH // 64, n_embd=WIDTH)
    recipe = bc.Recipe()
    recipe.lr, recipe.anvil_lr = recipe.lr * args.lr_mult, recipe.anvil_lr * args.lr_mult
    args.adapter_lr = args.adapter_lr or recipe.lr
    settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=args.route_k or bc.HIDDEN // 16,
                           cut_exact_keys=args.keys == 'exact', boost_exact_keys=args.boost_keys == 'exact')
    if args.chain_tokens:
        settings.chain_tokens = args.chain_tokens
    if args.key_chunk:
        settings.key_chunk = args.key_chunk
    kind = 'none' if args.credit_basis == 'full' else args.credit_basis
    args.credit_rank = min(args.credit_rank, WIDTH)
    args.adapter_rank = args.adapter_rank or args.credit_rank
    if args.update_mode not in ('dense', 'lora') and kind == 'none':
        raise ValueError(f'--update-mode {args.update_mode} needs a credit basis for its subspace.')
    if args.leakage_every and kind == 'none':
        raise ValueError('--leakage-every needs a credit basis to measure against.')
    diagnostic_only = kind == 'oracle-grad'
    if diagnostic_only and rank == 0:
        print('oracle-grad: full-cycle credit projected on its own history; a diagnostic bound, not a forward-only '
              'result', flush=True)

    def new_bank(basis_kind=kind):
        return SubspaceBank(basis_kind, WIDTH, n_layer, args.credit_rank, refresh=args.basis_refresh_steps,
                            warmup=args.basis_warmup_steps, explore=args.exploration_probe_fraction, explore_use=args.exploration_use,
                            extra_rank=args.adapter_rank, route_k=settings.route_k, seed=args.seed, device=device)

    Boost, Cut = make_estimators(bc)
    on_boost, on_cut = args.subspace_target in ('boost', 'both'), args.subspace_target in ('cut', 'both')

    batch1, batch2 = 4, 8
    if world % batch1 and batch1 % world or batch2 % world:
        raise ValueError('Run on 1, 2, 4 or 8 GPUs.')
    split = world // batch1 if batch1 < world else 1
    per1, per2 = max(1, batch1 // world), batch2 // world
    groups = [None]
    if split > 1:
        groups = [torch.distributed.new_group(list(range(g * split, (g + 1) * split))) for g in range(world // split)]
    group = groups[rank // split] if split > 1 else None

    training = bc.load_split(args.data_dir / args.train_file, args.train_rows)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(bc.SEED + 1))]
    val_rows = bc.load_split(args.data_dir / 'val.pt', VAL_ROWS)
    steps1, steps2 = len(training) // batch1, len(training) // batch2
    n1 = round(args.switch * steps1) // 2 * 2
    j0 = n1 * batch1 // batch2
    if args.steps:
        n1, last = min(n1, args.steps), j0 + args.steps
    else:
        last = steps2
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, preset=args.preset, model=asdict(config), seed=args.seed,
            switch=args.switch, boost_steps=n1, cut_steps=last - j0, boost_batch=batch1, cut_batch=batch2,
            train_file=args.train_file, train_rows=len(training), recipe=asdict(recipe), settings=asdict(settings),
            polar_margin=bc.POLAR_MARGIN, torch_version=torch.__version__, diagnostic_only=diagnostic_only,
            cgsc=dict(credit_basis=args.credit_basis, credit_rank=args.credit_rank,
                      basis_refresh_steps=args.basis_refresh_steps, basis_warmup_steps=args.basis_warmup_steps,
                      exploration_probe_fraction=args.exploration_probe_fraction,
                      exploration_use=args.exploration_use, subspace_target=args.subspace_target,
                      update_mode=args.update_mode, adapter_rank=args.adapter_rank,
                      adapter_merge_steps=args.adapter_merge_steps, adapter_lr=args.adapter_lr),
        ), indent=2, default=str) + '\n')

    def boost_init(bank):
        model = bc.make_model(device, args.seed, config)
        estimator = Boost(model, settings, args.seed, rank, world, split, group, bank=bank, enabled=on_boost)
        sites = estimator.writer_sites()
        return model, make_optimizer(model, recipe, steps1, args.update_mode, bank, sites, args), estimator

    # untimed warmup on random tokens: both phases on the full cycle, then (if a basis is used) on a random basis of
    # the run's rank, so every probe shape compiles before the clock starts; then training from a fresh model
    noise = torch.randint(0, config.vocab_size, (batch2, SEQ), generator=torch.Generator().manual_seed(bc.SEED)).to(device)
    first = (rank // split) * per1
    mine = slice(rank * per2, (rank + 1) * per2)
    for warm_kind in ['none'] + (['random'] if kind != 'none' else []):
        warm_bank = new_bank(warm_kind)
        if warm_kind == 'random':
            warm_bank.refresh()
        model = bc.make_model(device, args.seed, config)
        estimator = Boost(model, settings, args.seed, rank, world, split, group, bank=warm_bank, enabled=on_boost)
        estimator.step(noise[first:first + per1, :-1].contiguous(), noise[first:first + per1, 1:].contiguous(),
                       bc.make_optimizer(model, recipe, steps1))
        estimator.close()
        model = bc.make_model(device, args.seed, config)
        estimator = Cut(model, settings, args.seed, rank, world, bank=warm_bank, enabled=on_cut)
        estimator.step(noise[mine, :-1].contiguous(), noise[mine, 1:].contiguous(), bc.make_optimizer(model, recipe, steps2))
        estimator.close()
        del model, estimator
    if cuda:
        torch.cuda.empty_cache()
    bank = new_bank()
    model, optimizer, estimator = boost_init(bank)
    diagnostics = []

    def track(opt):
        _, writers, leakage = parts_of(opt)
        diagnostics.extend(x.diagnostics for x in (writers, leakage) if x is not None)

    track(optimizer)
    if rank == 0:
        print(f'cgsc {args.preset}: Boost {n1} steps at batch {batch1}, Cut {last - j0} steps at batch {batch2}; '
              f'basis {args.credit_basis} r={args.credit_rank}, update {args.update_mode}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    sync()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0

    def set_rate(step, steps):
        left = (steps - step + 1) / steps
        for g in optimizer.param_groups:
            g.setdefault('initial_lr', g['lr'])
            g['lr'] = g['initial_lr'] * min(1.0, left / recipe.warmdown)

    def save_state(tag):
        _, writers, _ = parts_of(optimizer)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / f'{tag}.pt')
        torch.save(dict(bank=bank.state_dict(), writers=writers.state_dict() if writers else None),
                   args.output / f'cgsc_{tag}.pt')

    def run_step(step, x, y):
        profiling = args.profile_every and step % args.profile_every == 0
        estimator.profile, estimator.times = bool(profiling), {}
        loss = estimator.step(x, y, optimizer)
        estimator.profile = False
        return loss, ({k: round(v, 4) for k, v in estimator.times.items()} if profiling else None)

    def after(phase, step, seqs, loss, times):
        nonlocal paused, elapsed
        sync()
        diag = sum(d.seconds for d in diagnostics)
        elapsed = time.perf_counter() - start - paused - diag
        record = dict(phase=phase, step=step, seqs=seqs, train_loss=loss, train_seconds=round(elapsed, 2),
                      diag_seconds=round(diag, 2))
        _, writers, leakage = parts_of(optimizer)
        extra = {**bank.pop_metrics(), **(writers.pop_metrics() if writers else {}),
                 **(leakage.pop_metrics() if leakage else {})}
        if extra:
            record['cgsc'] = {k: round(v, 5) if isinstance(v, float) else v for k, v in extra.items()}
        if times:
            record['profile'] = times
        if args.eval_every and seqs % args.eval_every == 0 and seqs < len(training):
            pause = time.perf_counter()
            record['val_loss'] = bc.evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        if args.save_every and seqs % args.save_every == 0:
            pause = time.perf_counter()
            if rank == 0:
                save_state(f'ckpt_{seqs}')
            if world > 1:
                torch.distributed.barrier()
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()

    for step in range(1, n1 + 1):
        first = (step - 1) * batch1 + (rank // split) * per1
        batch = training[first:first + per1].to(device)
        set_rate(step, steps1)
        loss, times = run_step(step, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        after('boost', step, step * batch1, loss, times)

    # the handoff: the same model and bank, the optimizer state under Cut's schedule at the same fraction of the
    # epoch; restricted writers merge and restart on Cut's sites (mlp.c_proj's credit is the next block's mid stream)
    estimator.close()
    old = optimizer
    estimator = Cut(model, settings, args.seed, rank, world, bank=bank, enabled=on_cut)
    optimizer = make_optimizer(model, recipe, steps2, args.update_mode, bank, estimator.writer_sites(), args)
    old_inner, old_writers, old_leakage = parts_of(old)
    new_inner, new_writers, new_leakage = parts_of(optimizer)
    for new_o, old_o in zip(new_inner.optimizers[:2], old_inner.optimizers[:2]):
        for p, state in old_o.state.items():
            new_o.state[p] = state
    new_inner.optimizers[0].t = j0
    if new_writers:
        for p, state in old_writers.fallback.state.items():
            new_writers.fallback.state[p] = state
        new_writers.fallback.t = j0
        new_writers.t, new_writers.merges = old_writers.t, old_writers.merges + old_writers.engaged
        new_writers.reference = old_writers.reference
    if new_leakage:
        new_leakage.t = old_leakage.t
    del old, old_inner, old_writers, old_leakage
    track(optimizer)
    for step in range(j0 + 1, last + 1):
        batch = training[(step - 1) * batch2:step * batch2][mine].to(device)
        set_rate(step, steps2)
        loss, times = run_step(step, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        after('cut', step, step * batch2, loss, times)

    estimator.close()
    result = dict(val_loss=bc.evaluate(model, val_rows), train_seconds=round(elapsed, 2), boost_steps=n1,
                  cut_steps=last - j0, tokens=last * batch2 * (SEQ - 1), diagnostic_only=diagnostic_only,
                  bases_generation=bank.generation)
    if args.test_file:
        result['test_loss'] = bc.evaluate(model, bc.load_split(args.data_dir / args.test_file))
    if rank == 0:
        save_state('final')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
