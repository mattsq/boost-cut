"""Boost -> Cut on a plain GPT (analysis only, never a Noise entry).

The benchmark's GPT has five pieces a plain GPT lacks: relu^2 MLPs, value embeddings (with their gates), the x0
injections and residual scalars, the logit soft cap, and half-sequence attention windows. The plain GPT here drops all
five: GELU MLPs (the tanh form, as in GPT-2), no value embeddings, a plain residual stream, an uncapped head, and full
causal attention in every block. It keeps the pre-norm blocks, the parameter-free RMS norm, rotary positions, QK-norm,
the untied head, the widths and the init.

patch() swaps the architecture-specific pieces of train/boostcut/run.py for their plain forms: the model, the head's
loss in every closed-form probe (no cap), the MLP differences (GELU instead of relu^2), Cut's routes (units ranked by
|GELU| times their output norm), and no value embeddings or mixing scalars to measure. Nothing else of the method
changes. analysis/backprop_preset.py --arch plain trains the same plain GPT with backprop.

    torchrun --standalone --nproc_per_node=8 analysis/plain_gpt.py --preset 38m --output /data/runs/plain_bc38
"""
import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model import GPT, MLP, GPTConfig, apply_rotary_emb, attention, norm  # noqa: E402

BC = None  # the patched train/boostcut/run.py module (its WIDTH and EPS are read when the kernels compile)


def gelu(u):
    return F.gelu(u, approximate='tanh')


class GeluMLP(MLP):
    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))


class PlainGPT(GPT):
    """GELU MLPs, no value embeddings, a plain residual stream, an uncapped head, full causal attention."""

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        for block in self.transformer.h:
            block.mlp = GeluMLP(config)
            block.attn.ve_gate = None
        self.value_embeds = nn.ModuleDict()
        self.window_sizes = [(config.sequence_len, 0)] * config.n_layer
        # the residual scalars become constants (1 and 0), kept as buffers so the method's shared code reads them
        del self.resid_lambdas, self.x0_lambdas
        self.register_buffer('resid_lambdas', torch.ones(config.n_layer), persistent=False)
        self.register_buffer('x0_lambdas', torch.zeros(config.n_layer), persistent=False)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.zero_()

    def forward(self, idx, targets=None, loss_reduction='mean'):
        T = idx.shape[1]
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        for i, block in enumerate(self.transformer.h):
            x = block(x, None, cos_sin, self.window_sizes[i])
        logits = self.lm_head(norm(x))[..., :self.config.vocab_size].float()
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                               reduction=loss_reduction)


# ------------------------------------------------- run.py's closed-form kernels, uncapped and with GELU differences

@torch.compile
def coordinate_differences(raw, y, sigma):
    z, up, down = raw.float(), raw.float() + sigma, raw.float() - sigma
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq):
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / BC.WIDTH + BC.EPS)
    clean = torch.logsumexp(base * inv0, -1, keepdim=True) - base_y * inv0
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / BC.WIDTH + BC.EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.logsumexp(z, -1) - (base_y + steps * dirs_y) * inv
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def row_derivatives(base, C, rows, y, steps, sq, dots, dir_sq):
    target = y.clamp(min=0)
    dirs = C[rows]
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = dirs.gather(2, target[:, None, None].expand(-1, rows.shape[1], 1)).squeeze(-1)
    inv0 = torch.rsqrt(sq / BC.WIDTH + BC.EPS)
    clean = torch.logsumexp(base * inv0, -1, keepdim=True) - base_y * inv0
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / BC.WIDTH + BC.EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * dirs) * inv.unsqueeze(-1)
    moved = torch.logsumexp(z, -1) - (base_y + steps * dirs_y) * inv
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def hidden_scores(ut, st, Hk, cols, g):
    moved = st.unsqueeze(-1) * Hk
    change = gelu(ut.unsqueeze(1) + moved) - gelu(ut.unsqueeze(1) - moved)
    stream_change = torch.bmm(change.to(torch.bfloat16), cols)
    scored = torch.bmm(stream_change, g.to(torch.bfloat16).unsqueeze(-1)).squeeze(-1).float()
    return scored / (2 * st)


@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + BC.EPS), weight)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)


@torch.compile(dynamic=False)
def chain_chunk(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                target_logit_w, y):
    L = len(mids)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site
    for b in range(first, L):
        if b > first:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        dz = BC.norm_difference(mids[b], delta)
        dh = torch.bmm(dz, w_fc[b][idx[b]].transpose(1, 2))
        h = hs[b].unsqueeze(1)
        delta = delta + torch.bmm(gelu(h + dh) - gelu(h), w_out_t[b][idx[b]])
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / BC.WIDTH + BC.EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.logsumexp(z, -1) - zy
    inv0 = torch.rsqrt(sq / BC.WIDTH + BC.EPS)
    clean = torch.logsumexp(base * inv0, -1, keepdim=True) - base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


def patch(bc):
    """Turns a loaded train/boostcut/run.py module into its plain-GPT form, in place."""
    global BC
    # compilation imports the modules of the functions it inlines by name: both must resolve to these copies
    assert getattr(sys.modules.get(__name__), '__dict__', None) is globals(), \
        f'register this module as sys.modules[{__name__!r}] before loading it'
    BC = bc
    sys.modules[bc.__name__] = bc
    bc.GPT = PlainGPT
    bc.GPTConfig = functools.partial(GPTConfig, window_pattern='L')
    bc.has_ve = lambda layer_idx, n_layer: False
    for name in ('coordinate_differences', 'cycle_derivatives', 'row_derivatives', 'hidden_scores',
                 'readout_losses', 'chain_chunk'):
        setattr(bc, name, globals()[name])

    class PlainBoost(bc.Boost):
        def stage_losses(self, *args):  # no mixing scalars to measure
            return torch.zeros((), device=self.device)

    class PlainCut(bc.Cut):
        @torch.no_grad()
        def estimate(self, x, y):
            """Cut.estimate with GELU routes: each token's units ranked by |GELU(h)| times their output norm."""
            d, model, s = self.capture, self.model, self.s
            if self.profile:
                torch.cuda.synchronize()
                self.last = bc.time.perf_counter()
            d.capture(x, y)
            self.tick('forward')
            L = model.config.n_layer
            b, t = x.shape
            yf = y.flatten()
            gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
            e = d.outputs['transformer.wte'].float().flatten(0, 1)
            x0 = norm(e)
            mids, outs, routes = [], [], []
            for i in range(L):
                p = f'transformer.h.{i}.'
                mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
                mlp = model.transformer.h[i].mlp
                hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
                r = gelu(hpre)
                outs.append(mid + F.linear(r, mlp.c_proj.weight.float()))
                top = (r.abs() * mlp.c_proj.weight.float().norm(dim=0)).topk(s.route_k, dim=-1).indices
                mids.append(mid), routes.append((top, hpre.gather(1, top)))
            self.tick('routes')
            g_top = self.probe(outs[-1], yf)
            self.tick('top')
            chain = self.chain_credits(e, x0, mids, outs, routes, yf)
            self.tick('chain')
            targets = {}
            for i in range(L):
                p = f'transformer.h.{i}.'
                block = model.transformer.h[i]
                d_mid, d_out = chain[i], chain[i + 1] if i < L - 1 else g_top
                gradients[p + 'attn.c_proj.weight'] = d_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
                gradients[p + 'mlp.c_proj.weight'] = d_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
                u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
                hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
                gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
                targets[i] = (d_mid @ block.attn.c_proj.weight.float()).view(b, t, bc.WIDTH)
            self.measured = dict(chain=chain, g_top=g_top, targets=targets)
            self.tick('writers_hidden')
            self.internals(gradients, targets, b, t, s.cut_exact_keys)
            self.tick('local')
            gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, bc.WIDTH))
            for name, p in model.named_parameters():
                if name not in gradients:
                    gradients[name] = torch.zeros_like(p, dtype=torch.float32)
            return gradients

    bc.Boost, bc.Cut = PlainBoost, PlainCut
    return bc


# ------------------------------------------------------- the Dust record (train/val/run.py) on the plain GPT

@torch.compile
def dust_head_loss(x, weight, targets, vocab):
    """The record's head_loss without the soft cap."""
    logits = F.linear(norm(x), weight)[..., :vocab].float()
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='none')


@torch.compile
def dust_block_forward(x, w, ve, cos, sin, window, o=None, attn=None, fc=None, mlp=None):
    """The record's block_forward (a block with optional jitter at o, attn, fc or mlp) for the plain block: GELU,
    no value embedding."""
    add = lambda t, jitter: t if jitter is None else t + jitter
    B, T, C = x.shape
    h = norm(x)
    dim = 2 * cos.shape[-1]
    q, k, v = (F.linear(h, w[name]).view(B, T, -1, dim) for name in ('q', 'k', 'v'))
    q, k = norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))
    y = add(attention(q, k, v, window).reshape(B, T, C), o)
    x = x + add(F.linear(y, w['proj']), attn)
    hidden = add(F.linear(norm(x), w['fc']), fc)
    return x + add(F.linear(gelu(hidden), w['out']), mlp)


def patch_dust(rec):
    """Turns a loaded train/val/run.py (the Dust record) into its plain-GPT form, in place: an uncapped head and
    coordinate head, GELU blocks, every hidden unit jittered (GELU has no dead units, so the record's live-unit mask
    would drop real credit), and no value embeddings, gates or mixing scalars to estimate. The estimator, population,
    draw counts and optimizer are the record's."""
    assert getattr(sys.modules.get(__name__), '__dict__', None) is globals(), \
        f'register this module as sys.modules[{__name__!r}] before loading it'
    sys.modules[rec.__name__] = rec
    rec.GPT = PlainGPT
    rec.has_ve = lambda layer_idx, n_layer: False
    rec.head_loss, rec.block_forward, rec.coordinate_differences = dust_head_loss, dust_block_forward, \
        coordinate_differences

    class PlainDUST(rec.DUST):
        def direct_errors(self, sites, draws, sigma, entry=None, gamma=0.0, retrieved=None):
            """The record's direct_errors with every hidden unit live."""
            retrieved = retrieved or {}
            priors = {s: self.alpha[s] * r for s, r in retrieved.items() if s in self.alpha}
            n, b, t = rec.chunk_size(draws, self.recipe.nrep), self.batch, self.length
            width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
            errors = {s: torch.zeros(b, t, width(s), device=self.device) for s in sites}
            for chunk in range(draws // n):
                noise = {s: self.orthogonal_noise(s, draws, chunk, n, b, t, width(s)).flatten(0, 1) for s in sites}
                perturbed_loss = self.rerun(n, entry, {s: sigma * a for s, a in noise.items()})
                reward = self.clean_loss.unsqueeze(0) - perturbed_loss
                if n > 1:
                    reward = reward - reward.mean(0, keepdim=True)
                if gamma:
                    reward = rec.reward_to_go(reward, gamma)
                if priors:
                    predicted = sigma * sum(torch.einsum('nbtd,btd->nbt', noise[s].view(n, b, t, -1), mu.to(self.dtype))
                                            for s, mu in priors.items()).float()
                    reward = reward + (predicted - predicted.mean(0, keepdim=True) if n > 1 else predicted)
                for site, a in noise.items():
                    errors[site] -= torch.einsum('nbt,nbtd->btd', reward.to(self.dtype),
                                                 a.view(n, self.batch, self.length, -1)).float() / sigma
            kept = 1 - 1 / n if n > 1 else 1.0
            errors = {s: error / draws + kept * priors.get(s, 0) for s, error in errors.items()}
            self.estimates.update({s: e for s, e in errors.items() if s in self.prior_sites})
            for site, r in retrieved.items():
                target = errors[site] / kept
                self.fits[site] = torch.stack([(r * target).sum(), (r * r).sum()])
            return errors

        def scalar_gradients(self, draws, sigma=0.03):  # no mixing scalars
            return {}

    rec.DUST = PlainDUST
    return rec


def load_boostcut():
    spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    bc = patch(load_boostcut())
    bc.main()  # run.py's trainer, unchanged, on the plain GPT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=Path('logs/boostcut'))
    output = parser.parse_known_args()[0].output
    if int(os.environ.get('RANK', 0)) == 0 and (output / 'config.json').exists():
        config = json.loads((output / 'config.json').read_text())
        config.update(arch='plain', wrapper=__file__)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
