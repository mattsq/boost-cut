"""Stage 0 of docs/proposals/circuit-guided-low-rank-credit.md (analysis only; autograd is the oracle here and never
trains anything): do subspaces built from forward-observable statistics capture the credit learning needs?

For each checkpoint, bases are built per residual-stream site (train/boostcut/subspace.py's sites and builders) from
--basis-rows training sequences, and scored on credit from --heldout-rows other sequences:
- the credit targets, all by autograd: 'cut' (Cut's estimand: attention frozen, MLPs restricted to each token's
  top route_k units), 'full' (the true gradient of the model's loss) and 'boost' (each stream's own readout loss);
- the bases: random, activation-pca, contrib, route-cov, oracle-grad (leading directions of the same target's
  credit on the basis rows: the upper bound), layer-shuffled (route-cov from another block) and time-shuffled
  (route-cov from the previous checkpoint in the list), and the held-out credit's own leading directions (in-sample);
- scores: rho = sum |Q^T g|^2 / sum |g|^2, the mean per-token cos(g, P g), and for 'cut' the signed agreement
  cos(P g_cut, g_full) (mean and fraction positive);
- a one-step test at --onestep-rank: each selected writer moves by an equal-norm step along -G, -P G (each basis, P
  of the writer's output site) and the loss change is measured on the held-out rows (G the full-loss gradient on the
  basis rows);
- basis overlap (mean squared principal cosine) between consecutive checkpoints.
Gate A is summarized at the end: at r <= W/8, each forward-only basis's mean 'cut' rho against random's, and the
share of sites at rho >= 0.8.

    python analysis/subspace_alignment.py --preset 38m --ckpts /data/runs/bc38f_s0/ckpt_4880.pt,... \\
        --output results/subspace_alignment_s0.jsonl
    python analysis/subspace_alignment.py --preset 2m --ckpts init --random-tokens --seq-len 64   # smoke test
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train/boostcut'))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = sys.modules['boostcut'] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
import subspace  # noqa: E402
from subspace import SubspaceBank, stream_sites  # noqa: E402
from utils.model import norm  # noqa: E402

FORWARD_KINDS = ('random', 'activation-pca', 'contrib', 'route-cov')


def graph(model, x, y, mode, route_k, params=()):
    """Credits {site: (b * t, W)} of the summed loss by autograd. mode 'cut': attention outputs held at their
    clean values and each MLP restricted to the token's top route_k units (the rest held), Cut's estimand; 'full':
    the model's own loss. With params, also returns their gradients."""
    T = x.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    frozen = mode == 'cut'
    with torch.enable_grad():
        leaves = [p.requires_grad_(True) for p in params]
        e = model.transformer.wte(x).float().detach().requires_grad_(True)
        x0 = norm(e)
        h, streams = x0, {'emb': e}
        lam_r, lam_0 = model.resid_lambdas.float().detach(), model.x0_lambdas.float().detach()
        for i, block in enumerate(model.transformer.h):
            xin = lam_r[i] * h + lam_0[i] * x0
            ve = model.value_embeds[str(i)](x).float() if str(i) in model.value_embeds else None
            a = block.attn(norm(xin.detach() if frozen else xin), ve, cos_sin, model.window_sizes[i]).float()
            mid = xin + (a.detach() if frozen else a)
            w_fc, w_out = block.mlp.c_fc.weight.float(), block.mlp.c_proj.weight.float()
            r = F.relu(F.linear(norm(mid), w_fc)).square()
            if frozen and route_k < w_fc.shape[0]:
                top = (r * w_out.norm(dim=0)).topk(route_k, dim=-1).indices
                mask = torch.zeros_like(r).scatter_(-1, top, 1.0)
                r = r * mask + (r * (1 - mask)).detach()
            h = mid + F.linear(r, w_out)
            for site, z in ((f'mid{i}', mid), (f'out{i}', h)):
                z.retain_grad()
                streams[site] = z
        logits = 15 * torch.tanh(F.linear(norm(h), model.lm_head.weight.float())[..., :model.config.vocab_size] / 15)
        F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction='sum').backward()
        grads = [p.grad.float().clone() for p in leaves]
        for p in leaves:
            p.grad = None
            p.requires_grad_(False)
    credits = {site: z.grad.flatten(0, 1).float() for site, z in streams.items()}
    return (credits, grads) if params else credits


@torch.no_grad()
def forward_statistics(model, x):
    """The streams and MLP activations the training run's bank observes: {site: (n, W)}, [relu(hpre)^2], [c_proj]."""
    T = x.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    e = model.transformer.wte(x).float()
    x0 = norm(e)
    h, streams, acts, writers = x0, {'emb': e.flatten(0, 1)}, [], []
    for i, block in enumerate(model.transformer.h):
        xin = model.resid_lambdas[i].float() * h + model.x0_lambdas[i].float() * x0
        ve = model.value_embeds[str(i)](x).float() if str(i) in model.value_embeds else None
        mid = xin + block.attn(norm(xin), ve, cos_sin, model.window_sizes[i]).float()
        a = F.relu(F.linear(norm(mid), block.mlp.c_fc.weight.float())).square()
        h = mid + F.linear(a, block.mlp.c_proj.weight.float())
        streams[f'mid{i}'], streams[f'out{i}'] = mid.flatten(0, 1), h.flatten(0, 1)
        acts.append(a.flatten(0, 1))
        writers.append(block.mlp.c_proj.weight)
    return streams, acts, writers


@torch.no_grad()
def readout_credit(model, streams, y):
    """Boost's estimand at each block stream: the gradient of the shared head's loss reading that stream."""
    weight = model.lm_head.weight.float()[:model.config.vocab_size]
    out = {}
    for site, z in streams.items():
        if site == 'emb':
            continue
        with torch.enable_grad():
            leaf = z.detach().clone().requires_grad_(True)
            bc.readout_losses(leaf, weight, y.flatten()).sum().backward()
        out[site] = leaf.grad
    return out


@torch.no_grad()
def held_loss(model, x, y):
    T = x.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    x0 = norm(model.transformer.wte(x).float())
    h = x0
    for i, block in enumerate(model.transformer.h):
        h = model.resid_lambdas[i].float() * h + model.x0_lambdas[i].float() * x0
        ve = model.value_embeds[str(i)](x).float() if str(i) in model.value_embeds else None
        h = h + block.attn(norm(h), ve, cos_sin, model.window_sizes[i]).float()
        h = h + block.mlp(norm(h)).float()
    logits = 15 * torch.tanh(F.linear(norm(h), model.lm_head.weight.float())[..., :model.config.vocab_size] / 15)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction='none')


def loss_change(new, old, y):
    """Mean per-token loss change, differenced token by token and summed in float64 (a step this small moves the
    mean loss by less than its float32 resolution)."""
    keep = y.flatten() >= 0
    return (new[keep].double() - old[keep].double()).mean().item()


def build_bases(model, x, y, kinds, rank, route_k, width, n_layer, seed, oracle_credit, device):
    """{kind: {site: Q (W, rank)}} from the basis rows, by the training run's own builders."""
    streams, acts, writers = forward_statistics(model, x)
    valid = y.flatten() >= 0
    bases = {}
    for kind in kinds:
        bank = SubspaceBank(kind, width, n_layer, rank, warmup=1, route_k=route_k, seed=seed, device=device)
        if kind == 'random':
            bank.refresh()
        elif kind == 'oracle-grad':
            for site, g in oracle_credit.items():
                bank.observe_credit(site, g[valid])
            bank.end_step()
        else:
            bank.observe_forward(streams, acts, writers, valid)
            bank.end_step()
        bases[kind] = dict(bank.Q)
    return bases


def scores(Q, G, G_full=None):
    P = lambda g: (g @ Q) @ Q.T
    keep = G.norm(dim=1) > 0
    out = dict(rho=round(subspace.capture(Q, G), 5),
               cos=round(((G[keep] @ Q).norm(dim=1) / G[keep].norm(dim=1)).mean().item(), 5))
    if G_full is not None:
        agree = F.cosine_similarity(P(G)[keep], G_full[keep], dim=1)
        out.update(agree_full=round(agree.mean().item(), 5), agree_positive=round((agree > 0).float().mean().item(), 5))
    return out


def load_rows(path, rows, offset, seq_len, random_tokens, vocab, seed):
    if random_tokens:
        data = torch.randint(0, vocab, (rows, (seq_len or 64) + 1), generator=torch.Generator().manual_seed(seed))
    else:
        data = bc.load_split(path)[offset:offset + rows]
        if seq_len:
            data = data[:, :seq_len + 1]
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--preset', choices=sorted(bc.PRESETS), default='38m')
    parser.add_argument('--ckpts', required=True, help='Comma-separated state dicts in training order; "init" is '
                        'the fresh model (--seed).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--basis-rows', type=int, default=4, help='Training rows the bases are built from.')
    parser.add_argument('--basis-offset', type=int, default=0)
    parser.add_argument('--heldout-file', default='val.pt')
    parser.add_argument('--heldout-rows', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=0, help='Truncate the rows (0: the full 2,048 tokens).')
    parser.add_argument('--ranks', default='8,16,32,64,128,256')
    parser.add_argument('--route-k', type=int, default=0, help='Cut\'s routes (0: hidden/16).')
    parser.add_argument('--onestep-rank', type=int, default=0, help='Rank of the one-step test (0: W/8).')
    parser.add_argument('--onestep-blocks', default='', help='Blocks whose writers take the one-step test '
                        '(default: first, middle, last).')
    parser.add_argument('--step-norm', type=float, default=1e-2, help='One-step size relative to |W|_F.')
    parser.add_argument('--random-tokens', action='store_true', help='Smoke test on random tokens.')
    parser.add_argument('--output', type=Path, default=Path('logs/subspace_alignment.jsonl'))
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    else:
        torch._dynamo.config.disable = True  # a CPU smoke test: eager
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    W = bc.WIDTH
    route_k = args.route_k or bc.HIDDEN // 16
    ranks = [r for r in map(int, args.ranks.split(',')) if r <= W]
    top_rank = max(ranks)
    r1 = args.onestep_rank or W // 8
    blocks = [int(b) for b in args.onestep_blocks.split(',')] if args.onestep_blocks else sorted({0, n_layer // 2,
                                                                                                    n_layer - 1})
    config = bc.GPTConfig(n_layer=n_layer, n_head=W // 64, n_kv_head=W // 64, n_embd=W)  # the run's windows, even
    # when the rows are truncated
    xb, yb = load_rows(args.data_dir / 'train.pt', args.basis_rows, args.basis_offset, args.seq_len,
                       args.random_tokens, config.vocab_size, args.seed + 11)
    xh, yh = load_rows(args.data_dir / args.heldout_file, args.heldout_rows, 0, args.seq_len, args.random_tokens,
                       config.vocab_size, args.seed + 12)
    xb, yb, xh, yh = (z.to(device) for z in (xb, yb, xh, yh))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open('w')

    def emit(row):
        out.write(json.dumps(row) + '\n')
        out.flush()

    sites = stream_sites(n_layer)
    previous, summary = None, []
    for label in args.ckpts.split(','):
        model = bc.make_model(device, args.seed, config)
        if label != 'init':
            model.load_state_dict(torch.load(label, map_location=device))
        tag = 'init' if label == 'init' else Path(label).stem
        # credits on the basis rows (the oracle's history) and on the held-out rows (the scores)
        basis_credit = dict(cut=graph(model, xb, yb, 'cut', route_k))
        streams_b, _, _ = forward_statistics(model, xb)
        basis_credit['boost'] = readout_credit(model, streams_b, yb)
        held = dict(cut=graph(model, xh, yh, 'cut', route_k), full=graph(model, xh, yh, 'full', route_k))
        streams_h, _, _ = forward_statistics(model, xh)
        held['boost'] = readout_credit(model, streams_h, yh)
        valid = yh.flatten() >= 0
        bases = build_bases(model, xb, yb, FORWARD_KINDS, top_rank, route_k, W, n_layer, args.seed, {}, device)
        oracles = {target: build_bases(model, xb, yb, ['oracle-grad'], top_rank, route_k, W, n_layer, args.seed,
                                       basis_credit.get(target, basis_credit['cut']), device)['oracle-grad']
                   for target in ('cut', 'boost')}
        oracles['full'] = oracles['cut']  # the full gradient is never available to training: its bound is Cut's
        shuffled = {site: bases['route-cov'][sites[(sites.index(site) + 2) % len(sites)]] for site in sites}
        for target, credit in held.items():
            for site, G in credit.items():
                G = G[valid]
                G_full = held['full'][site][valid] if target == 'cut' else None
                U = torch.linalg.svd(G.double(), full_matrices=False)[2].T.float()  # in-sample leading directions
                candidates = {**{k: bases[k][site] for k in FORWARD_KINDS}, 'oracle-grad': oracles[target][site],
                              'layer-shuffled': shuffled[site], 'in-sample': U}
                if previous is not None:
                    candidates['time-shuffled'] = previous['route-cov'][site]
                for kind, Q in candidates.items():
                    for r in ranks:
                        row = dict(ckpt=tag, target=target, site=site, kind=kind, rank=r,
                                   **scores(Q[:, :r], G, G_full))
                        emit(row)
                        summary.append(row)
        # the one-step test: writers of the chosen blocks, equal-norm steps, loss change on the held-out rows
        names = [f'transformer.h.{b}.{k}.c_proj.weight' for b in blocks for k in ('attn', 'mlp')]
        named = dict(model.named_parameters())
        _, grads = graph(model, xb, yb, 'full', route_k, params=[named[n] for n in names])
        base_loss = held_loss(model, xh, yh)
        for name, G in zip(names, grads):
            block, kind_w = int(name.split('.')[2]), name.split('.')[3]
            site = f'mid{block}' if kind_w == 'attn' else f'out{block}'
            p = named[name]
            original = p.detach().clone()
            size = args.step_norm * max(original.float().norm().item(), 1e-3)
            directions = {'unprojected': G, **{k: bases[k][site][:, :r1] @ (bases[k][site][:, :r1].T @ G)
                                               for k in FORWARD_KINDS},
                          'oracle-grad': oracles['cut'][site][:, :r1] @ (oracles['cut'][site][:, :r1].T @ G)}
            for kind, D in directions.items():
                with torch.no_grad():
                    p.copy_((original.float() - size * D / D.norm().clamp_min(1e-30)).to(p.dtype))
                change = loss_change(held_loss(model, xh, yh), base_loss, yh)
                with torch.no_grad():
                    p.copy_(original)
                emit(dict(ckpt=tag, onestep=name, site=site, kind=kind, rank=r1, step_norm=args.step_norm,
                          loss_change=change, gradient_capture=round((D.norm() / G.norm()).item() ** 2, 5)))
        if previous is not None:
            for kind in ('activation-pca', 'contrib', 'route-cov'):
                for site in sites:
                    for r in ranks:
                        emit(dict(ckpt=tag, previous=previous['tag'], overlap_kind=kind, site=site, rank=r,
                                  overlap=round(subspace.overlap(bases[kind][site][:, :r],
                                                                 previous[kind][site][:, :r]), 5)))
        previous = {**bases, 'tag': tag}
        print(f'{tag}: done', flush=True)

    # Gate A: at r <= W / 8, forward-only bases against random on Cut's estimand
    gate = {}
    for row in summary:
        if row['target'] != 'cut' or row['rank'] > W // 8:
            continue
        key = (row['ckpt'], row['kind'], row['rank'])
        gate.setdefault(key, []).append(row['rho'])
    lines = []
    for (ckpt, kind, r), rhos in sorted(gate.items()):
        random_mean = sum(gate[ckpt, 'random', r]) / len(gate[ckpt, 'random', r])
        mean = sum(rhos) / len(rhos)
        lines.append(dict(gate_a=True, ckpt=ckpt, kind=kind, forward_only=kind in FORWARD_KINDS, rank=r,
                          mean_rho=round(mean, 4),
                          random_rho=round(random_mean, 4), over_random=round(mean / max(random_mean, 1e-12), 3),
                          sites_at_0_8=round(sum(x >= 0.8 for x in rhos) / len(rhos), 3)))
    for line in lines:
        emit(line)
        print(json.dumps(line), flush=True)
    out.close()


if __name__ == '__main__':
    main()
