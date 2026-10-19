"""Boost and Cut with residual-stream credit measured in a per-site subspace (circuit-guided subspace credit; see
docs/proposals/circuit-guided-low-rank-credit.md and subspace.py).

At a site whose SubspaceBank basis is active, a probe moves each token's stream along the basis's r directions q_j
(scaled to the Hadamard rows' norm, |b_j|^2 = WIDTH, so a move is as large as the full cycle's) instead of a full
cycle of WIDTH directions, and the credit is sum_j d_j q_j = P g, the projection of the credit the full cycle
measures. Everything else is the original estimator's: Cut's route selection, frozen attention, head credit,
hidden_credit(), the queries, keys, values and gates, the mixing scalars, and the optimizer. Boost's hub probes (in
the attention output's space) keep the full cycle. With the subspace off (kind 'none', a phase it does not target,
or before a site's first basis) every probe is the original code path, bit for bit.

The bank also gathers its forward statistics here, from the clean pass every step, and refreshes between steps, so
both are timed as part of training.
"""
import math
import time

import torch


def make_estimators(bc):
    """The CGSC estimators over a loaded train/boostcut/run.py module `bc` (its WIDTH is read at call time)."""

    class SubspaceCredit:
        """Mixin over bc.Boost / bc.Cut: probes at active sites along the bank's basis."""

        def __init__(self, *args, bank, enabled, **kwargs):
            super().__init__(*args, **kwargs)
            self.bank, self.enabled, self.queue = bank, enabled, []

        def probe_sites(self):
            raise NotImplementedError

        def writer_sites(self):
            raise NotImplementedError

        def readout_derivatives(self, x, y, dirs):
            """Forward differences (n, P) of each token's readout loss along stream directions dirs (P, WIDTH); the
            GPUs of a sequence each take a slice of the directions and add their parts."""
            weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
            stream_dirs = dirs.T.contiguous()
            C = (weight @ stream_dirs).T.contiguous()
            dir_sq = stream_dirs.square().sum(0)
            steps = self.s.sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            n, P, vocab = x.shape[0], dirs.shape[0], C.shape[1]
            lo, hi = self.part * P // self.split, (self.part + 1) * P // self.split
            derivative = torch.zeros(n, P, device=self.device)
            if hi > lo:
                per = max(1, self.s.chunk // ((hi - lo) * vocab))
                for t0 in range(0, n, per):
                    xs, ys = x[t0:t0 + per], y[t0:t0 + per]
                    derivative[t0:t0 + per, lo:hi] = bc.cycle_derivatives(
                        xs @ weight.T, C[lo:hi].contiguous(), ys, steps[t0:t0 + per],
                        xs.square().sum(-1, keepdim=True), xs @ stream_dirs[:, lo:hi], dir_sq[lo:hi])
            return self.group_sum(derivative)

        def probe(self, x, y, mix=None, probed=None):
            site = self.queue.pop(0) if self.queue else None
            bank = self.bank
            if not self.enabled or site is None or mix is not None:
                return super().probe(x, y, mix, probed)
            if bank.kind == 'oracle-grad' or not bank.active(site):
                g = super().probe(x, y, mix, probed)
                bank.observe_credit(site, g)
                return bank.project(site, g) if bank.active(site) else g
            self.basis()  # the full path's draws, so the random streams stay those of the original code
            torch.randint(0, bc.WIDTH, (x.shape[0], 1), device=self.device, generator=self.generator)
            rows, sentinels = bank.directions(site)
            dirs = torch.cat([rows, sentinels]) * math.sqrt(bc.WIDTH)
            return bank.credit(site, self.readout_derivatives(x, y, dirs), rows, sentinels)

        @torch.no_grad()
        def observe(self, y):
            """The bank's forward statistics from this step's clean pass."""
            bank, d, model = self.bank, self.capture, self.model
            if not bank.needs_forward:
                return
            L = model.config.n_layer
            valid = y.flatten() >= 0
            streams, activations, writers = {}, [], []
            if bank.kind == 'activation-pca':
                streams['emb'] = d.outputs['transformer.wte'].float().flatten(0, 1)
                for i in range(L):
                    p = f'transformer.h.{i}.'
                    mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
                    streams[f'mid{i}'] = mid
                    streams[f'out{i}'] = mid + d.outputs[p + 'mlp.c_proj'].float().flatten(0, 1)
            else:
                for i in range(L):
                    activations.append(d.inputs[f'transformer.h.{i}.mlp.c_proj'].flatten(0, 1))
                    writers.append(model.transformer.h[i].mlp.c_proj.weight)
            bank.observe_forward(streams, activations, writers, valid)

        @torch.no_grad()
        def estimate(self, x, y):
            timing = self.profile and not isinstance(self, bc.Cut)  # Cut starts its own clock
            if timing:
                torch.cuda.synchronize()
                self.last = time.perf_counter()
            self.queue = self.probe_sites()
            gradients = super().estimate(x, y)
            self.queue = []
            if timing:
                self.tick('estimate')
            self.observe(y)
            self.tick('subspace_stats')
            return gradients

        def step(self, x, y, optimizer):
            loss = super().step(x, y, optimizer)
            self.tick('optimizer')
            self.bank.end_step()
            self.tick('subspace_refresh')
            return loss

    class SubspaceBoost(SubspaceCredit, bc.Boost):
        def probe_sites(self):
            return [f'{kind}{i}' for i in range(self.model.config.n_layer) for kind in ('mid', 'hub', 'out')]

        def writer_sites(self):
            L = self.model.config.n_layer
            return {**{f'transformer.h.{i}.attn.c_proj.weight': f'mid{i}' for i in range(L)},
                    **{f'transformer.h.{i}.mlp.c_proj.weight': f'out{i}' for i in range(L)}}

    class SubspaceCut(SubspaceCredit, bc.Cut):
        def __init__(self, model, settings, seed, rank, world, bank, enabled):
            super().__init__(model, settings, seed, rank, world, bank=bank, enabled=enabled)

        def probe_sites(self):
            return [f'out{self.model.config.n_layer - 1}']  # the top stream, through the head alone

        def writer_sites(self):
            L = self.model.config.n_layer
            return {**{f'transformer.h.{i}.attn.c_proj.weight': f'mid{i}' for i in range(L)},
                    **{f'transformer.h.{i}.mlp.c_proj.weight': f'mid{i + 1}' if i < L - 1 else f'out{L - 1}'
                       for i in range(L)}}

        def chain_credits(self, e, x0, mids, outs, routes, y):
            """Cut's chain credits with each active site's moves along its basis (r directions plus sentinels)
            instead of a full cycle; the mixing scalars' four moves stay appended after them."""
            if not self.enabled:
                return super().chain_credits(e, x0, mids, outs, routes, y)
            model, s, bank, W = self.model, self.s, self.bank, bc.WIDTH
            L = model.config.n_layer
            vocab = model.config.vocab_size
            weight = model.lm_head.weight.float()[:vocab]
            weight_t = weight.T.to(torch.bfloat16).contiguous()
            w_fc = [blk.mlp.c_fc.weight.float() for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.float().T.contiguous() for blk in model.transformer.h]
            lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
            xL = outs[-1]
            base = xL @ weight.T
            target_w = weight[y.clamp(min=0)]
            n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
            credits = {}
            for site in [-1] + list(range(L)):
                name = 'emb' if site < 0 else f'mid{site}'
                H = self.basis()  # drawn at every site, so the random stream is the full path's
                measured = bank.active(name) and bank.kind != 'oracle-grad'
                if measured:
                    rows, sentinels = bank.directions(name)
                    D = torch.cat([rows, sentinels]) * math.sqrt(W)
                else:
                    D = H
                P = D.shape[0]
                probed = e if site < 0 else mids[site]
                steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                derivative_all = torch.empty(n, P, device=self.device)
                lam = torch.zeros(2, device=self.device)
                for t0 in range(0, n, m):
                    sl = slice(t0, t0 + m)
                    move = steps[sl].unsqueeze(-1) * D.unsqueeze(0)
                    scale = steps[sl].expand(-1, P)
                    if site >= 0:  # the block's mixing scalars, central differences, after the P moves
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1), -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h), torch.full_like(scale[:, :2], -h)], 1)
                    derivative = bc.chain_chunk(
                        site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, w_fc, w_out_t, weight_t, base[sl], target_w[sl], y[sl])
                    derivative_all[sl] = derivative[:, :P]
                    if site >= 0:
                        lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                if measured:
                    credit = bank.credit(name, derivative_all, rows, sentinels)
                else:
                    credit = derivative_all @ H / W
                    bank.observe_credit(name, credit)
                    if bank.active(name):  # oracle-grad: the full cycle, projected
                        credit = bank.project(name, credit)
                credits['emb' if site < 0 else site] = credit
                if site >= 0:
                    credits['lam', site] = lam
            return credits

    return SubspaceBoost, SubspaceCut


def clean_gradients(estimator, x, y):
    """Run an estimate with autograd unavailable: any .backward() or autograd.grad raises (forward-only check)."""
    def refuse(*args, **kwargs):
        raise RuntimeError('autograd called in forward-only training')

    saved = torch.Tensor.backward, torch.autograd.grad, torch.autograd.backward
    torch.Tensor.backward = torch.autograd.grad = torch.autograd.backward = refuse
    try:
        gradients = estimator.estimate(x, y)
    finally:
        torch.Tensor.backward, torch.autograd.grad, torch.autograd.backward = saved
    if any(g.requires_grad for g in gradients.values()):
        raise RuntimeError('an estimate carries an autograd graph')
    return gradients

