"""Rank-restricted updates of the residual writers (attn.c_proj and mlp.c_proj of every block), stage 2 of
docs/proposals/circuit-guided-low-rank-credit.md.

Each writer's measured gradient G (its p.grad after the ranks' reduction, built by Capture.reassemble from the
projected output credit) updates the writer under one of these modes:
- 'dense': the record's optimizer (ANVIL II) on W, unrestricted; UpdateLeakage measures how far its steps leave the
  credit subspace.
- 'projected-sgd': W <- W - lr Q Q^T G, plain SGD that keeps every step inside the subspace (the geometry without
  the optimizer's transform). Q follows the bank's current basis.
- 'qa': W = W_base + Q A with Q fixed for the whole run and A trained by AdamW on G_A = Q^T G (no autograd).
- 'qa-merge': the same, and every merge_steps updates Q A is merged into W_base, A and its AdamW state are reset to
  zero, and Q is re-read from the bank, so the cumulative change can exceed rank r.
- 'lora': W = W_base + B A, two factors trained by AdamW on G_B = G A^T and G_A = B^T G (A random, B zero at each
  start), merged like 'qa-merge' when merge_steps > 0.
The forward pass keeps the full matrix: W itself is rewritten as W_base + Q A after each update, so the model, its
hooks and the evaluation are unchanged. The adapters engage once every writer has a basis (the end of the bank's
warmup); until then the writers take the dense optimizer's steps.
"""
import time

import torch

MODES = ('dense', 'projected-sgd', 'qa', 'qa-merge', 'lora')


def writer_names(n_layer):
    return [f'transformer.h.{i}.{kind}.c_proj.weight' for i in range(n_layer) for kind in ('attn', 'mlp')]


def effective_rank(M):
    """exp of the entropy of the normalized singular values (1 for rank one, the rank for a flat spectrum)."""
    s = torch.linalg.svdvals(M.double())
    p = s / s.sum().clamp_min(1e-30)
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


class Diagnostics:
    """Wall time spent on diagnostics only (synchronized), for the run to subtract from training time."""

    def __init__(self):
        self.seconds = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - self.start


class WriterUpdates:
    """The writers' optimizer under a restricted mode (see the module docstring). Exposes param_groups (the lr
    schedule scales them) and step(); the fallback optimizer updates the writers until the adapters engage."""

    def __init__(self, model, bank, mode, rank, sites, lr, betas=(0.9, 0.95), merge_steps=0, fallback=None,
                 track_rank=True, seed=0):
        if mode not in MODES or mode == 'dense':
            raise ValueError(f'WriterUpdates takes a restricted mode, not {mode!r}.')
        self.model, self.bank, self.mode, self.rank = model, bank, mode, rank
        self.merge_steps, self.fallback, self.track_rank = merge_steps, fallback, track_rank
        named = dict(model.named_parameters())
        self.params = {name: named[name] for name in writer_names(model.config.n_layer)}
        self.sites = dict(sites)
        self.engaged, self.t, self.merges = False, 0, 0
        self.Q, self.base, self.reference = {}, {}, {}
        self.diagnostics = Diagnostics()
        self.metrics = {}
        generator = torch.Generator().manual_seed(seed + 31)
        self.A, self.B, factors = {}, {}, []
        for name, p in self.params.items():
            out_dim, in_dim = p.shape
            if mode in ('qa', 'qa-merge', 'lora'):
                self.A[name] = torch.zeros(rank, in_dim, device=p.device, dtype=torch.float32)
                factors.append(self.A[name])
            if mode == 'lora':
                self.B[name] = torch.zeros(out_dim, rank, device=p.device, dtype=torch.float32)
                factors.append(self.B[name])
        self.generator = generator
        if factors:
            self.inner = torch.optim.AdamW(factors, lr=lr, betas=betas, eps=1e-8, weight_decay=0.0)
            self.param_groups = self.inner.param_groups
        else:
            self.inner = None
            self.param_groups = [dict(params=[], lr=lr)]
        if fallback is not None:
            self.param_groups = self.param_groups + list(fallback.param_groups)

    # ---------------------------------------------------------------------------------------------------- engage

    def needs_basis(self):
        return self.mode != 'lora'

    def _start(self, name):
        """A fresh adapter on the writer's current value: W_base = W, A = 0 (lora: A random, B = 0)."""
        p = self.params[name]
        self.base[name] = p.detach().float().clone()
        if self.mode in ('qa', 'qa-merge'):
            self.A[name].zero_()
        if self.mode == 'lora':
            in_dim = p.shape[1]
            bound = 1 / in_dim ** 0.5
            self.A[name].copy_(torch.empty(self.A[name].shape).uniform_(-bound, bound, generator=self.generator))
            self.B[name].zero_()

    def engage(self):
        for name in self.params:
            if self.needs_basis():
                self.Q[name] = self.bank.basis(self.sites[name], self.rank).clone()
            self._start(name)
            if self.track_rank and name not in self.reference:  # kept across the handoff's restart
                self.reference[name] = self.params[name].detach().float().clone()
        self.engaged = True

    def set_sites(self, sites):
        """The writers' output sites change at the handoff (Cut measures mlp.c_proj's credit at the next block's mid
        stream): merge and restart on the new sites' bases."""
        self.sites = dict(sites)
        if self.engaged and self.mode != 'lora':
            self.merge()

    # ------------------------------------------------------------------------------------------------------ step

    def zero_grad(self, set_to_none=True):
        if self.fallback is not None:
            self.fallback.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        if not self.engaged:
            if not self.needs_basis() or self.bank.ready(set(self.sites.values())):
                self.engage()
            elif self.fallback is not None:
                self.fallback.step()
                return
            else:
                raise RuntimeError('The writers have no basis yet and no fallback optimizer.')
        lr = self.param_groups[0]['lr']
        for name, p in self.params.items():
            G = p.grad.float()
            if self.mode == 'projected-sgd':
                Q = self.bank.basis(self.sites[name], self.rank)
                p.copy_(p.float() - lr * (Q @ (Q.T @ G)))
                continue
            if self.mode == 'lora':
                self.A[name].grad = self.B[name].T @ G
                self.B[name].grad = G @ self.A[name].T
            else:
                self.A[name].grad = self.Q[name].T @ G
        if self.inner is not None:
            self.inner.step()
            self.inner.zero_grad(set_to_none=True)
            for name, p in self.params.items():
                left = self.B[name] if self.mode == 'lora' else self.Q[name]
                p.copy_(self.base[name] + left @ self.A[name])
        self.t += 1
        if self.merge_steps and self.mode in ('qa-merge', 'lora') and self.t % self.merge_steps == 0:
            self.merge()

    @torch.no_grad()
    def merge(self):
        """W_base <- W_base + Q A (the writer already holds it), restart A and its AdamW state, re-read Q."""
        for name, p in self.params.items():
            if self.mode in ('qa', 'qa-merge') and name in self.Q:
                self.Q[name] = self.bank.basis(self.sites[name], self.rank).clone()
            self._start(name)
        if self.inner is not None:
            self.inner.state.clear()
        self.merges += 1
        self.metrics['merges'] = self.merges
        if self.track_rank and self.reference:
            with self.diagnostics:
                ranks = [effective_rank(p.float() - self.reference[n]) for n, p in self.params.items()]
            self.metrics['cumulative_effective_rank'] = sum(ranks) / len(ranks)

    # -------------------------------------------------------------------------------------------------- state

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics

    def state_dict(self):
        cpu = lambda d: {k: v.cpu() for k, v in d.items()}
        return dict(mode=self.mode, rank=self.rank, t=self.t, merges=self.merges, engaged=self.engaged,
                    sites=dict(self.sites), Q=cpu(self.Q), base=cpu(self.base), reference=cpu(self.reference),
                    A=cpu(self.A), B=cpu(self.B), generator=self.generator.get_state(),
                    inner=self.inner.state_dict() if self.inner is not None else None,
                    fallback=self.fallback.state_dict() if self.fallback is not None else None)

    def load_state_dict(self, state):
        if (state['mode'], state['rank']) != (self.mode, self.rank):
            raise ValueError(f'Checkpoint adapters are {state["mode"]} rank {state["rank"]}.')
        device = next(iter(self.params.values())).device
        to = lambda d: {k: v.to(device) for k, v in d.items()}
        self.t, self.merges, self.engaged, self.sites = state['t'], state['merges'], state['engaged'], state['sites']
        self.Q, self.base, self.reference = to(state['Q']), to(state['base']), to(state['reference'])
        for store, saved in ((self.A, state['A']), (self.B, state['B'])):
            for name, value in saved.items():
                store[name].copy_(value)  # in place: the AdamW instance holds these tensors
        self.generator.set_state(state['generator'])
        if self.inner is not None:
            self.inner.load_state_dict(state['inner'])
        if self.fallback is not None and state['fallback'] is not None:
            self.fallback.load_state_dict(state['fallback'])


class UpdateLeakage:
    """Wraps the run's optimizer and measures, every `every` updates, how much of each writer's actual step leaves its
    output site's credit subspace: |(I - P) dW| / |dW|, averaged over the writers. Untimed (Diagnostics)."""

    def __init__(self, optimizer, model, bank, sites, every):
        self.optimizer, self.bank, self.sites, self.every = optimizer, bank, dict(sites), every
        named = dict(model.named_parameters())
        self.params = {name: named[name] for name in self.sites}
        self.param_groups = optimizer.param_groups
        self.diagnostics = Diagnostics()
        self.t, self.metrics = 0, {}

    def __getattr__(self, name):  # the run reaches through to .optimizers at the handoff
        return getattr(self.__dict__['optimizer'], name)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.t += 1
        due = self.every and self.t % self.every == 0 and self.bank.ready(set(self.sites.values()))
        if due:
            with self.diagnostics:
                before = {name: p.detach().float().clone() for name, p in self.params.items()}
        self.optimizer.step()
        if due:
            with self.diagnostics:
                shares = []
                for name, p in self.params.items():
                    delta = p.float() - before[name]
                    Q = self.bank.basis(self.sites[name])
                    outside = delta - Q @ (Q.T @ delta)
                    shares.append((outside.norm() / delta.norm().clamp_min(1e-30)).item())
                self.metrics['update_leakage'] = sum(shares) / len(shares)

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics
