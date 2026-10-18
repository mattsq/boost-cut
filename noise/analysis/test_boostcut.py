"""Checks train/boostcut/run.py's measured credits against autograd, at any model size (analysis only).

Cut: chain credits at the raw embedding and every mid stream, the top-stream probe, the mixing scalars and the exact
keys, against autograd of the same estimand (the route-restricted attention-cut loss; the keys' local step against
the same targets). Boost: its exact keys against autograd of their local step. Reports one estimate's time split.

    python analysis/test_boostcut.py --preset 38m --ckpt /data/runs/hybrid_cutk_f0.3/final.pt
    python analysis/test_boostcut.py --preset 7m
    python analysis/test_boostcut.py --preset 38m --arch plain     # analysis/plain_gpt.py's GPT
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from utils.model import norm  # noqa: E402
from utils.prepare_data import TRAIN_ROWS  # noqa: E402

cosine = lambda a, b: round(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item(), 4)
ratio = lambda a, b: round((a.norm() / b.norm()).item(), 4)
PLAIN = None  # analysis/plain_gpt.py, loaded by --arch plain
act = lambda u: PLAIN.gelu(u) if PLAIN else F.relu(u).square()
cap = lambda z: z if PLAIN else 15 * torch.tanh(z / 15)


def cut_reference(model, x, y, route_k):
    """Autograd of the route-restricted attention-cut loss at the raw embedding, the mid streams, the top stream and
    the mixing scalars (float32, summed loss)."""
    T = x.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    with torch.enable_grad():
        e = model.transformer.wte(x).float().detach().requires_grad_(True)
        lam_r = model.resid_lambdas.float().detach().requires_grad_(True)
        lam_0 = model.x0_lambdas.float().detach().requires_grad_(True)
        x0 = norm(e)
        h_, mids = x0, []
        for i, block in enumerate(model.transformer.h):
            xin = lam_r[i] * h_ + lam_0[i] * x0
            ve = model.value_embeds[str(i)](x).float() if str(i) in model.value_embeds else None
            a = block.attn(norm(xin.detach()), ve, cos_sin, model.window_sizes[i]).float()
            mid = xin + a.detach()
            mid.retain_grad()
            mids.append(mid)
            w_fc, w_out = block.mlp.c_fc.weight.float(), block.mlp.c_proj.weight.float()
            r = act(F.linear(norm(mid), w_fc))
            if route_k < w_fc.shape[0]:
                top = (r.abs() * w_out.norm(dim=0)).topk(route_k, dim=-1).indices
                mask = torch.zeros_like(r).scatter_(-1, top, 1.0)
                r = r * mask + (r * (1 - mask)).detach()
            h_ = mid + F.linear(r, w_out)
        h_.retain_grad()
        logits = cap(F.linear(norm(h_), model.lm_head.weight.float())[..., :model.config.vocab_size])
        F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction='sum').backward()
    return dict(emb=e.grad[0], mids=[m.grad[0] for m in mids], top=h_.grad[0], lam=torch.stack([lam_r.grad, lam_0.grad]))


def key_reference(model, est, targets, i):
    """Autograd of the keys' local step at layer i: sum over queries of target . attention output, keys moving."""
    cfg = model.config
    d = est.capture
    T, h, hd = d.length, cfg.n_head, cfg.n_embd // cfg.n_head
    cos, sin = model.cos[0, :T, 0].float(), model.sin[0, :T, 0].float()
    rot = lambda z: torch.cat([z[..., :hd // 2] * cos + z[..., hd // 2:] * sin, z[..., :hd // 2] * -sin + z[..., hd // 2:] * cos], -1)
    unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + bc.EPS)
    pre = f'transformer.h.{i}.attn.'
    heads = lambda name: d.outputs[pre + name][0].float().view(T, h, hd).transpose(0, 1)
    q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
    if str(i) in model.value_embeds:
        ve = d.outputs[f'value_embeds.{i}'][0].float().view(T, h, hd).transpose(0, 1)
        gate = 2 * torch.sigmoid(d.outputs[pre + 've_gate'][0].float().view(T, h)).T.unsqueeze(-1)
        values = values + gate * ve
    pos = torch.arange(T, device=q_raw.device)
    lag = pos[:, None] - pos[None, :]
    ok = (lag >= 0) & (lag <= model.window_sizes[i][0])
    hub = targets[i][0].view(T, h, hd).transpose(0, 1)
    with torch.enable_grad():
        k_leaf = k_raw.clone().requires_grad_(True)
        attn = (unit(rot(q_raw)) @ unit(rot(k_leaf)).transpose(1, 2) / hd ** 0.5).masked_fill(~ok, float('-inf')).softmax(-1)
        (hub * (attn @ values)).sum().backward()
    return k_leaf.grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--preset', default='38m')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--ks', default='full,16', help='Route budgets: full and/or hidden divisors.')
    parser.add_argument('--chain-tokens', type=int, default=0)
    parser.add_argument('--key-chunk', type=int, default=0)
    parser.add_argument('--arch', choices=('noise', 'plain'), default='noise')
    args = parser.parse_args()
    global PLAIN
    if args.arch == 'plain':
        spec_p = importlib.util.spec_from_file_location('plain_gpt', ROOT / 'analysis/plain_gpt.py')
        PLAIN = sys.modules['plain_gpt'] = importlib.util.module_from_spec(spec_p)
        spec_p.loader.exec_module(PLAIN)
        PLAIN.patch(bc)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0)
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    config = bc.GPTConfig(n_layer=n_layer, n_head=bc.WIDTH // 64, n_kv_head=bc.WIDTH // 64, n_embd=bc.WIDTH)
    model = bc.make_model(device, 0, config)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    rows = bc.load_split(args.data_dir / 'train.pt', TRAIN_ROWS)[:1].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    out = {}
    for route_k in [bc.HIDDEN if k == 'full' else bc.HIDDEN // int(k) for k in args.ks.split(',')]:
        settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=route_k, boost_exact_keys=True)
        if args.chain_tokens:
            settings.chain_tokens = args.chain_tokens
        if args.key_chunk:
            settings.key_chunk = args.key_chunk
        est = bc.Cut(model, settings, 0, 0, 1)
        est.estimate(x, y)  # compile
        est.profile, est.times = True, {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        est.estimate(x, y)
        seconds = time.perf_counter() - start
        est.profile = False
        meas, ref = est.measured, cut_reference(model, x, y, route_k)
        L = n_layer
        res = dict(seconds=round(seconds, 3), parts={k: round(v, 4) for k, v in est.times.items()},
                   emb=cosine(meas['chain']['emb'], ref['emb']), top=cosine(meas['g_top'], ref['top']),
                   mids=[cosine(meas['chain'][i], ref['mids'][i]) for i in range(L)],
                   lam=[cosine(torch.stack([meas['chain']['lam', i] for i in range(L)], 1), ref['lam']),
                        ratio(torch.stack([meas['chain']['lam', i] for i in range(L)], 1), ref['lam'])],
                   keys={i: [cosine(est.key_credit[i][0], k_ref := key_reference(model, est, meas['targets'], i)),
                             ratio(est.key_credit[i][0], k_ref)] for i in sorted({0, L // 2, L - 1})})
        out[f'cut_k{route_k}'] = res
        print(json.dumps({f'cut_k{route_k}': res}), flush=True)
        est.close()
    # Boost with exact keys: its keys against autograd of their local step from Boost's measured hub targets
    settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=bc.HIDDEN // 16, boost_exact_keys=True)
    est = bc.Boost(model, settings, 0, 0, 1)
    est.estimate(x, y)
    est.profile, est.times = True, {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    est.estimate(x, y)
    seconds = time.perf_counter() - start
    # Boost keeps no targets: recompute them the way estimate did is not needed; check keys on fresh hub probes
    d = est.capture
    targets = {}
    for i in range(n_layer):
        p = f'transformer.h.{i}.'
        xm = (d.blocks[i] + d.outputs[p + 'attn.c_proj']).float().flatten(0, 1)
        att_in = d.inputs[p + 'attn.c_proj'].flatten(0, 1)
        targets[i] = est.probe(xm, y.flatten(), mix=model.transformer.h[i].attn.c_proj.weight, probed=att_in).view(1, -1, bc.WIDTH)
    est.exact_key_errors(targets)
    res = dict(seconds=round(seconds, 3), keys={i: [cosine(est.key_credit[i][0], k_ref := key_reference(model, est, targets, i)),
                                                    ratio(est.key_credit[i][0], k_ref)] for i in sorted({0, n_layer // 2, n_layer - 1})})
    out['boost_exact_keys'] = res
    print(json.dumps({'boost_exact_keys': res}), flush=True)
    est.close()


if __name__ == '__main__':
    main()
