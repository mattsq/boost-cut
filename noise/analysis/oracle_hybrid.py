"""Two-phase schedules priced with exact credit (analysis only: autograd measures what the measured methods could
reach, it never trains an entry).

Phase 1 is Boost's stagewise objective (oracle_boost.local_forward: sub-block stages, the embedding at the first
attention stage along its own path, the head on every readout, the mixing scalars at their stage) at batch 4, as
in the measured Boost run. Phase 2 is one of
  attn  the attention-cut gradient: every attention reads a detached stream, so credit reaches a block only through
        later MLPs, the identity path and the x0 injections (Cut's estimand)
  full  backprop, the estimand of Dust
at batch 8. The epoch is the record's 4,880 sequences in the record's order: phase 1 takes the first fraction f of
them in batches of 4, phase 2 the rest in batches of 8. The weights and the optimizer state (ANVIL's rails, Adam's
moments) are handed over; each phase's learning rates and ANVIL schedule are those of a one-phase run at the same
fraction of the epoch, so a handoff changes the objective and the batch, nothing else.

Phase 1 runs once, with snapshots at every switch point. f = 1 (Boost alone) and f = 0 (phase 2 alone) are priced
too, with --ends.

    python analysis/oracle_hybrid.py --out /data/runs/oracle_hybrid --switches 0.3,0.5,0.7 --phase2 attn,full --ends
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('oracle_boost', ROOT / 'analysis/oracle_boost.py')
oracle_boost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_boost)
bc = oracle_boost.bc
from utils.model import GPTConfig  # noqa: E402
from utils.eval import evaluate  # noqa: E402
from utils.model import norm  # noqa: E402
from utils.prepare_data import TRAIN_ROWS, VAL_ROWS, load_rows  # noqa: E402

BOOST = types.SimpleNamespace(unit='sub', emb='block0own', head='all', scalars='local')


ROUTE = {'k': 0}  # > 0: credit passes through each token's top-k relu^2 units per block only


def routed_mlp(mlp, z, k):
    """The MLP with the stream's credit restricted to each token's k largest units (by output contribution): the
    other units keep their values but pass no credit down, as if held at their clean outputs. Every unit's weights
    still learn, one local step from the block-output credit, as in the measured methods."""
    w_fc, w_out = mlp.c_fc.weight, mlp.c_proj.weight
    r = F.relu(F.linear(z.detach(), w_fc)).square()  # the values; c_fc learns from every unit through this path
    r_z = F.relu(F.linear(z, w_fc.detach())).square()  # the same values, carrying credit down to the stream
    mask = torch.zeros_like(r).scatter_(-1, (r * w_out.norm(dim=0)).detach().topk(k, dim=-1).indices, 1.0)
    return F.linear(r + (r_z * mask - (r_z * mask).detach()), w_out)


def cut_forward(model, idx, targets):
    """The final loss with every attention's input detached (the attention-cut credit graph)."""
    T = idx.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    x = norm(model.transformer.wte(idx))
    x0 = x
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        a = block.attn(norm(x.detach()), ve, cos_sin, model.window_sizes[i])
        mid = x + a
        x = mid + (routed_mlp(block.mlp, norm(mid), ROUTE['k']) if ROUTE['k'] else block.mlp(norm(mid)))
    logits = model.lm_head(norm(x))[..., :model.config.vocab_size].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def gradients(model, x, y, objective):
    """Set every parameter's .grad to the exact gradient of the objective; returns the final readout's loss."""
    model.requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if objective == 'boost':
            final, loss = oracle_boost.local_forward(model, x, y, BOOST)
        elif objective == 'attn':
            final = loss = cut_forward(model, x, y)
        else:
            final = loss = model(x, y)
    loss.backward()
    model.requires_grad_(False)
    for p in model.parameters():  # the optimizers read float gradients, as the estimators hand them
        g = torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float()
        p.grad = g.to(p.dtype)
    return final.item()


def set_rate(optimizer, step, steps, warmdown):
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * min(1.0, left / warmdown)


def snapshot(model, optimizer):
    """The weights and each parameter's optimizer state, by name."""
    names = {id(p): n for n, p in model.named_parameters()}
    state = {}
    for o in optimizer.optimizers:
        for p, st in o.state.items():
            state[names[id(p)]] = {k: v.clone() if torch.is_tensor(v) else v for k, v in st.items()}
    return dict(weights={k: v.clone() for k, v in model.state_dict().items()}, state=state)


def restore(model, snap, recipe, steps, t):
    """The snapshot's weights, and an optimizer for a `steps`-update schedule at update t with its state."""
    model.load_state_dict(snap['weights'])
    optimizer = bc.make_optimizer(model, recipe, steps)
    anvil, adam = optimizer.optimizers
    in_anvil = {id(p) for g in anvil.param_groups for p in g['params']}
    for name, p in model.named_parameters():
        if name in snap['state']:
            target = anvil if id(p) in in_anvil else adam
            target.state[p] = {k: v.clone() if torch.is_tensor(v) else v for k, v in snap['state'][name].items()}
    anvil.t = t
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--switches', default='0.3,0.5,0.7', help='Fractions of the epoch trained by phase 1.')
    parser.add_argument('--phase2', default='attn,full')
    parser.add_argument('--ends', action='store_true', help='Also price f = 1 (Boost alone) and f = 0 (phase 2 alone).')
    parser.add_argument('--polar-margin', type=float, default=1.1)
    parser.add_argument('--seed', type=int, default=bc.SEED)
    parser.add_argument('--route-k', type=int, default=0, help='Phase 2 (attn): credit through each token\'s '
                        'top-k MLP units only.')
    parser.add_argument('--stage-temp', action='store_true', help='Phase 1: a learned temperature per '
                        'intermediate readout (objective only).')
    parser.add_argument('--data-frac', type=float, default=1.0, help='Train on this fraction of the epoch, '
                        'with the whole schedule compressed into it.')
    args = parser.parse_args()
    ROUTE['k'] = args.route_k
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0)
    bc.POLAR_MARGIN = args.polar_margin
    recipe = bc.Recipe()
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(bc.SEED + 1))]
    training = training[:round(len(training) * args.data_frac) // 8 * 8]
    steps1, steps2 = len(training) // 4, len(training) // 8  # 1,220 and 610 for the whole epoch
    switches = sorted(float(f) for f in args.switches.split(','))
    at = {round(f * steps1) // 2 * 2: f for f in switches}  # phase 1 ends on a phase-2 batch boundary
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    log = (args.out / 'metrics.jsonl').open('a')
    results = {}

    def record(**entry):
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + '\n')
        log.flush()

    probe_rows = load_rows(args.data_dir / 'val.pt', VAL_ROWS)[:1].to(device)

    @torch.no_grad()
    def knob(model):
        """The top stream's largest coordinate: its median share of each token's squared norm, and its peak."""
        seen = {}
        hook = model.transformer.h[-1].register_forward_hook(lambda m, a, o: seen.setdefault('top', o.float()))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(probe_rows[:, :-1].contiguous())
        hook.remove()
        top = seen['top'].flatten(0, 1)
        return round((top.square().amax(-1) / top.square().sum(-1)).median().item(), 4), round(top.abs().max().item())

    def finish(name, model):
        results[name] = evaluate({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.data_dir)
        share, peak = knob(model)
        record(run=name, val_loss=results[name], knob_share=share, knob_max=peak,
               seconds=round(time.perf_counter() - start, 1))
        (args.out / 'result.json').write_text(json.dumps(results, indent=2) + '\n')

    start = time.perf_counter()
    model = bc.make_model(device, args.seed, GPTConfig())
    optimizer = bc.make_optimizer(model, recipe, steps1)
    temp_opt = None
    if args.stage_temp:
        oracle_boost.STAGE['temps'] = torch.zeros(32, device=device, requires_grad=True)
        temp_opt = torch.optim.Adam([oracle_boost.STAGE['temps']], lr=0.01, betas=(0.9, 0.95))
    snaps = {}
    last = steps1 if args.ends else max(at)
    for step in range(1, last + 1):
        set_rate(optimizer, step, steps1, recipe.warmdown)
        batch = training[(step - 1) * 4:step * 4].to(device)
        loss = gradients(model, batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), 'boost')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if temp_opt is not None:
            temp_opt.step()
            temp_opt.zero_grad(set_to_none=True)
        if step % 61 == 0:
            extra = {} if temp_opt is None else dict(temps=[round(t, 2) for t in oracle_boost.STAGE['temps'].tolist()[:16]])
            record(run='boost', step=step, train_loss=loss, **extra)
        if step in at:
            snaps[at[step]] = snapshot(model, optimizer)
    if args.ends:
        finish('boost_f1.0', model)

    for kind in args.phase2.split(','):
        for f in ([0.0] if args.ends else []) + switches:
            first = next((s1 for s1, g in at.items() if g == f), 0) // 2 + 1
            if f == 0:
                model = bc.make_model(device, args.seed, GPTConfig())
                optimizer = bc.make_optimizer(model, recipe, steps2)
            else:
                optimizer = restore(model, snaps[f], recipe, steps2, first - 1)
            for step in range(first, steps2 + 1):
                set_rate(optimizer, step, steps2, recipe.warmdown)
                batch = training[(step - 1) * 8:step * 8].to(device)
                loss = gradients(model, batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), kind)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % 61 == 0:
                    record(run=f'{kind}_f{f}', step=step, train_loss=loss)
            finish(f'{kind}_f{f}', model)


if __name__ == '__main__':
    main()
