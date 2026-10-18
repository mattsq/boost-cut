"""Does the "already clean" picture (Cammarata, "Neural Networks Are Already Clean", Goodfire 2026) hold for networks
trained without backprop, and does it make credit cheaper to measure? Analysis only: autograd computes the exact
credits that the measurements are compared against.

For each checkpoint, on held-out sequences:
  sparsity   per token and block, how many relu^2 units carry 50/90/99% of the MLP output (unit i's share:
             relu^2(h_i) |W_out[:, i]|), and the live fraction (h_i > 0)
  ablation   val loss when each token's relu^2 outputs below a fraction f of its largest are zeroed
             ("weak activations don't matter")
  attention  per query, the number of keys holding 90% of the attention mass, and the mass on edges >= 0.1 max
  knob       share of the stream's squared norm carried by its largest coordinate (the essay's "saliency" dimension)
  routes     the attention-cut credit at every stream site with gradients only through each token's top-k MLP units
             per block (the other units held at their clean outputs), against the full attention-cut credit: per-token
             cosine, and weight-level cosine of the MLP gradients one local step from those credits
  prior      per-token cosine of each site's attention-cut credit with the top-stream credit

    python analysis/clean_probe.py --ckpts /data/runs/bp_recordopt/final.pt,... --names bp,... --seqs 4
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.eval import load_model  # noqa: E402
from utils.model import apply_rotary_emb, norm  # noqa: E402
from utils.prepare_data import VAL_ROWS, load_rows  # noqa: E402


def forward(model, idx, y, cut=False, route_k=0, weak_frac=0.0, rec=None):
    """The model's function in float32. cut: every attention reads a detached stream (the attention-cut credit
    graph). route_k: gradients pass through each token's top-k relu^2 units per block only; the others keep their
    values but are held (detached). weak_frac: relu^2 outputs below this fraction of the token's largest are zeroed.
    rec: filled with the streams (for credit) and per-block internals. Returns the summed loss."""
    T = idx.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    vocab = model.config.vocab_size
    x = norm(model.transformer.wte(idx).float())
    if torch.is_grad_enabled():  # the embedding stream is the graph's leaf, so every site's credit can be read off
        x = x.detach().requires_grad_(True)
    x0 = x
    if rec is not None:
        rec['x0'] = x0
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx).float() if str(i) in model.value_embeds else None
        a = block.attn(norm(x.detach() if cut else x), ve, cos_sin, model.window_sizes[i])
        mid = x + a
        h = F.linear(norm(mid), block.mlp.c_fc.weight)
        r = F.relu(h).square()
        w_out = block.mlp.c_proj.weight
        contrib = r * w_out.norm(dim=0)
        if weak_frac:
            r = r * (contrib >= weak_frac * contrib.amax(-1, keepdim=True))
        if route_k:
            mask = torch.zeros_like(r).scatter_(-1, contrib.topk(route_k, dim=-1).indices, 1.0)
            r = r * mask + (r * (1 - mask)).detach()
        out = mid + F.linear(r, w_out)
        if rec is not None:
            rec[f'in{i}'], rec[f'mid{i}'], rec[f'out{i}'] = x, mid, out
            rec[f'h{i}'], rec[f'contrib{i}'] = h.detach(), contrib.detach()
        x = out
    logits = 15 * torch.tanh(model.lm_head(norm(x))[..., :vocab].float() / 15)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction='sum')


def site_names(n_layer):
    return ['x0'] + [f'{kind}{i}' for i in range(n_layer) for kind in ('mid', 'out')]


def credits(model, x, y, route_k=0):
    """Attention-cut credit at every stream site (per token), and the per-block internals of that forward."""
    rec = {}
    model.requires_grad_(False)
    with torch.enable_grad():
        loss = forward(model, x, y, cut=True, route_k=route_k, rec=rec)
        names = site_names(model.config.n_layer)
        grads = torch.autograd.grad(loss, [rec[n] for n in names])
    return {n: g.detach() for n, g in zip(names, grads)}, rec


def mlp_grads(model, rec, cr):
    """Each block's MLP weight gradients one local step from the block-output credit (the way the measured methods
    train c_fc and c_proj): c_proj from credit (x) relu^2; c_fc from the credit through relu^2 at every unit."""
    out = {}
    for i, block in enumerate(model.transformer.h):
        d = cr[f'out{i}'].flatten(0, 1)
        h = rec[f'h{i}'].flatten(0, 1)
        r = F.relu(h).square()
        xin = norm(rec[f'mid{i}'].detach()).flatten(0, 1)
        g_h = (d @ block.mlp.c_proj.weight) * 2 * F.relu(h)
        out[f'c_proj{i}'] = d.T @ r
        out[f'c_fc{i}'] = g_h.T @ xin
    return out


def cos_rows(a, b):
    a, b = a.flatten(0, -2), b.flatten(0, -2)
    keep = (a.norm(dim=-1) > 0) & (b.norm(dim=-1) > 0)
    return F.cosine_similarity(a[keep], b[keep], dim=-1).mean().item()


def cos_all(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


@torch.no_grad()
def attention_stats(model, rec, seq=0):
    """Per layer: median over queries (and heads) of the keys holding 90% of the attention mass, and mean mass on
    edges with A >= 0.1 max, for one sequence."""
    stats = []
    T = rec['in0'].shape[1]
    cos, sin = model.cos[:, :T], model.sin[:, :T]
    pos = torch.arange(T, device=rec['in0'].device)
    lag = pos[:, None] - pos[None, :]
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        xn = norm(rec[f'in{i}'][seq:seq + 1])
        h, hd = attn.n_head, attn.head_dim
        q = norm(apply_rotary_emb(F.linear(xn, attn.c_q.weight).view(1, T, h, hd), cos, sin))[0].transpose(0, 1)
        k = norm(apply_rotary_emb(F.linear(xn, attn.c_k.weight).view(1, T, h, hd), cos, sin))[0].transpose(0, 1)
        window = model.window_sizes[i][0]
        logits = (q @ k.transpose(-1, -2)) / hd ** 0.5
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        A = logits.softmax(-1)  # (h, T, T)
        srt = A.sort(-1, descending=True).values
        k90 = (srt.cumsum(-1) < 0.9).sum(-1).float() + 1
        strong = (A * (A >= 0.1 * A.amax(-1, keepdim=True))).sum(-1)
        stats.append(dict(keys90=k90.median().item(), keys90_q75=k90.quantile(0.75).item(),
                          strong_mass=strong.mean().item()))
        del logits, A, srt
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--ckpts', required=True, help='Comma list of state-dict paths.')
    parser.add_argument('--names', required=True)
    parser.add_argument('--seqs', type=int, default=4)
    parser.add_argument('--route-ks', default='32,64,128,256,512')
    parser.add_argument('--weak-fracs', default='0.01,0.05,0.1,0.25')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda', 0)
    rows = load_rows(args.data_dir / 'val.pt', VAL_ROWS)[:args.seqs].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    n_tok = y.numel()
    results = {}
    for path, name in zip(args.ckpts.split(','), args.names.split(',')):
        model = load_model(torch.load(path, map_location=device), device).float()
        model.transformer.wte.float()
        for ve in model.value_embeds.values():
            ve.float()
        L = model.config.n_layer
        res = {}
        with torch.no_grad():
            rec = {}
            clean = forward(model, x, y, rec=rec).item() / n_tok
            res['loss'] = clean
            # sparsity: units carrying 50/90/99% of each token's MLP output, live fraction
            sp = []
            for i in range(L):
                c = rec[f'contrib{i}'].flatten(0, 1)
                share = c.sort(-1, descending=True).values.cumsum(-1) / c.sum(-1, keepdim=True).clamp_min(1e-30)
                ks = {f'k{int(q * 100)}': ((share < q).sum(-1).float() + 1).median().item() for q in (0.5, 0.9, 0.99)}
                ks['live'] = (rec[f'h{i}'] > 0).float().mean().item()
                sp.append(ks)
            res['sparsity'] = sp
            # the saliency knob: largest coordinate's share of the squared norm, at each block output
            knob = []
            for i in range(L):
                o = rec[f'out{i}'].flatten(0, 1)
                frac = (o.square().amax(-1) / o.square().sum(-1)).median().item()
                top = o.abs().argmax(-1).mode().values.item()
                knob.append(dict(share=frac, coord=top, max=o.abs().max().item()))
            res['knob'] = knob
            res['attention'] = attention_stats(model, rec)
            res['ablation'] = {f: forward(model, x, y, weak_frac=float(f)).item() / n_tok - clean
                               for f in args.weak_fracs.split(',')}
            del rec
        full, rec = credits(model, x, y)
        g_full = mlp_grads(model, rec, full)
        names = site_names(L)
        res['prior'] = {n: cos_rows(full[n], full[f'out{L - 1}']) for n in names}
        res['routes'] = {}
        for k in (int(k) for k in args.route_ks.split(',')):
            restricted, rec_k = credits(model, x, y, route_k=k)
            g_k = mlp_grads(model, rec_k, restricted)
            res['routes'][k] = dict(
                token_cos={n: cos_rows(restricted[n], full[n]) for n in names},
                weight_cos={n: cos_all(g_k[n], g_full[n]) for n in g_full},
            )
            del restricted, rec_k, g_k
        del full, rec, g_full, model
        torch.cuda.empty_cache()
        results[name] = res
        print(json.dumps({name: res}), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
