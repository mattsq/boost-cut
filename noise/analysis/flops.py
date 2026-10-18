"""Compute per training token for each method, counted (analysis only): matmul FLOPs from torch's FlopCounterMode,
with compilation disabled so every product is dispatched and counted, for one update of each method. Elementwise work
is not FLOPs in that count; the dominant elementwise cost of the forward-only methods, soft-capped cross-entropies
over the vocabulary, is counted separately as vocabulary entries evaluated per token.

    python analysis/flops.py --preset 38m --methods backprop,boost,cut,dust
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch._dynamo
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
from utils.model import GPTConfig  # noqa: E402
from utils.prepare_data import TRAIN_ROWS  # noqa: E402


def dense_attention(q, k, v, window):
    """utils.model.attention as plain matrix products, so the counter sees it (dense: an upper bound on the windowed
    kernel's work, applied alike to every method). Tensors are [batch, token, head, dim]."""
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))
    T = q.shape[2]
    pos = torch.arange(T, device=q.device)
    ok = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] <= window)
    att = ((q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5).masked_fill(~ok, float('-inf')).softmax(-1)
    return (att @ v.float()).to(q.dtype).transpose(1, 2)


def dense_flex(q, k, v, block_mask=None, return_lse=False):
    """flex attention with a window mask, as plain matrix products; block_mask is the window (see window_int)."""
    T = q.shape[2]
    pos = torch.arange(T, device=q.device)
    ok = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] <= block_mask)
    logits = ((q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5).masked_fill(~ok, float('-inf'))
    lse = torch.logsumexp(logits, -1)
    out = (torch.exp(logits - lse.unsqueeze(-1)) @ v.float()).to(q.dtype)
    return (out, lse) if return_lse else out


def patch(*modules):
    for m in modules:
        if hasattr(m, 'attention'):
            m.attention = dense_attention
        if hasattr(m, 'flex'):
            m.flex = dense_flex
        if hasattr(m, 'window_mask'):
            m.window_mask = lambda window, length, device: window


def count(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    torch.cuda.synchronize()
    return counter.get_total_flops()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--preset', default='38m')
    parser.add_argument('--methods', default='backprop,boost,cut,dust')
    args = parser.parse_args()
    torch._dynamo.config.disable = True  # eager: every product goes through the counter
    import utils.model as um
    patch(um, bc)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    L, W, V = n_layer, bc.WIDTH, 4096
    config = GPTConfig(n_layer=n_layer, n_head=W // 64, n_kv_head=W // 64, n_embd=W)
    rows = bc.load_split(args.data_dir / 'train.pt', TRAIN_ROWS)[:8].to(device)
    out = dict(preset=args.preset, params=sum(p.numel() for p in bc.make_model(device, 0, config).parameters()))
    tokens1 = 2048  # one sequence
    settings = bc.Settings(fc_k=bc.HIDDEN // 8, route_k=bc.HIDDEN // 16, cut_exact_keys=True)
    for method in args.methods.split(','):
        model = bc.make_model(device, 0, config)
        if method == 'backprop':
            x, y = rows[:1, :-1].contiguous(), rows[:1, 1:].contiguous()

            def fn():
                model.requires_grad_(True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(x, y).backward()
                model.requires_grad_(False)
            flops = count(fn) / tokens1
            out[method] = dict(flops_per_token=flops, vocab_evals_per_token=V)
        elif method in ('boost', 'cut'):
            est = bc.Boost(model, settings, 0, 0, 1) if method == 'boost' else bc.Cut(model, settings, 0, 0, 1)
            x, y = rows[:1, :-1].contiguous(), rows[:1, 1:].contiguous()
            flops = count(lambda: est.estimate(x, y)) / tokens1
            est.close()
            if method == 'boost':  # readouts: 2 per block but the last block's MLP (the final readout) + hubs
                readouts = 2 * L + L
                vocab = readouts * W * V + (2 * L - 1) * V + V  # probes, intermediate coordinate heads, final head
            else:
                vocab = (L + 1) * (W + 4) * V + W * V + V  # chain sites (with scalar moves), top probes, head
            out[method] = dict(flops_per_token=flops, vocab_evals_per_token=vocab)
        elif method == 'dust':
            spec_r = importlib.util.spec_from_file_location('record', ROOT / 'train/val/run.py')
            rec = importlib.util.module_from_spec(spec_r)
            spec_r.loader.exec_module(rec)
            patch(rec)
            recipe = rec.recipe(1.0)
            recipe.writers, recipe.hidden, recipe.hubs = (lst[:1] * L for lst in (recipe.writers, recipe.hidden, recipe.hubs))
            est = rec.DUST(model, recipe, 0, 0, 8)  # rank 0 of 8: one GPU's share of the population
            x, y = rows[:1, :-1].contiguous(), rows[:1, 1:].contiguous()  # its work is per token: one sequence
            optimizer = rec.make_optimizer(model, recipe, 610)
            flops = 8 * count(lambda: est.step(x, y, optimizer)) / tokens1
            est.close()
            out[method] = dict(flops_per_token=flops)
        print(json.dumps({method: out[method]}), flush=True)
    for method in args.methods.split(','):
        if method != 'backprop' and 'backprop' in out:
            out[method]['x_backprop'] = round(out[method]['flops_per_token'] / out['backprop']['flops_per_token'], 1)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
