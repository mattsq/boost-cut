"""Per-site probe bases for circuit-guided subspace credit (docs/proposals/circuit-guided-low-rank-credit.md).

A site is a residual-stream position: 'emb' (the raw embedding), 'mid{i}' (block i's stream after its attention) and
'out{i}' (block i's output). Boost probes mid{i} and out{i}; Cut probes emb, every mid{i} and out{L-1} (the top
stream). A SubspaceBank keeps, for each site, an orthonormal Q (W, R) built from statistics gathered between
refreshes, and a probe at an active site moves the stream along the leading columns of Q instead of a full cycle.
The measured credit is then P g = Q Q^T g (deterministic, biased by design), not an estimate of g.

Kinds of basis:
- 'none': no subspace; every probe is the original full cycle.
- 'random': a seeded Haar-random Q, redrawn at each refresh. Controls for rank alone.
- 'activation-pca': the leading eigenvectors of the site's streamed activation covariance (centered).
- 'contrib': the block's MLP output columns, each weighted by its summed token contribution relu(hpre)^2 |c_proj_j|:
  W_hat diag(sum_t a_tj |w_j|) W_hat^T, W_hat the unit columns.
- 'route-cov': the covariance of the actual output contributions v_tj = relu(hpre_tj)^2 c_proj[:, j] of each token's
  top route_k units (Cut's routes): W diag(sum_t a_tj^2 [j in top_t]) W^T.
- 'oracle-grad': the leading eigenvectors of the measured full-cycle credit's second moment. Every probe at an active
  site still runs a full cycle (its cost is not reduced) and is then projected: an upper bound, never a forward-only
  result.

Statistics are summed over the steps since the last refresh and over all ranks; the eigendecomposition runs on rank 0
and Q is broadcast. Q is held fixed within a step and only changes in end_step(), between optimizer steps.

Exploration (sentinels): a fraction of each active site's probe budget moves along random orthonormal directions in
the complement of the in-subspace columns, redrawn every step. Their responses estimate the credit energy the basis
misses. 'measure' only logs it; 'augment' also keeps their credit second moment and, at the next refresh, swaps the
basis's last columns for its leading directions; 'update' adds their rescaled response to the credit (the random
subset estimator of the missing part, (W - r) / m sum_j d_j u_j).
"""
import math

import torch

KINDS = ('none', 'random', 'activation-pca', 'contrib', 'route-cov', 'oracle-grad')
EXPLORE = ('measure', 'augment', 'update')


def sign_convention(Q):
    """Flip each column so that its largest-magnitude entry is positive (eigenvectors and QR factors are defined up
    to sign)."""
    pick = Q.abs().argmax(0)
    signs = torch.sign(Q.gather(0, pick.unsqueeze(0))).squeeze(0)
    return Q * torch.where(signs == 0, torch.ones_like(signs), signs)


def orthonormalize(M):
    """An orthonormal basis of the columns of M (W, r), in order, with the sign convention."""
    Q, R = torch.linalg.qr(M.double())
    Q = Q * torch.sign(torch.diagonal(R)).where(torch.diagonal(R) != 0, torch.ones(1, dtype=Q.dtype, device=Q.device))
    return sign_convention(Q).to(M.dtype)


def leading_eigenvectors(M, r):
    """The r leading eigenvectors (W, r) of a symmetric matrix, in descending order of eigenvalue, and all its
    eigenvalues (descending). Computed in float64."""
    values, vectors = torch.linalg.eigh(M.double())
    values, vectors = values.flip(0), vectors.flip(1)
    return sign_convention(vectors[:, :r]).float(), values.float()


def random_basis(width, r, seed, device):
    """A seeded Haar-random orthonormal basis (width, r); the same on every device and rank."""
    generator = torch.Generator().manual_seed(seed)
    return orthonormalize(torch.randn(width, r, generator=generator, dtype=torch.float64)).float().to(device)


def complement_sample(Q, m, seed):
    """m random orthonormal directions (W, m) orthogonal to the columns of Q (W, r)."""
    generator = torch.Generator().manual_seed(seed)
    G = torch.randn(Q.shape[0], m, generator=generator, dtype=torch.float64).to(Q.device)
    Qd = Q.double()
    G = G - Qd @ (Qd.T @ G)
    G = G - Qd @ (Qd.T @ G)  # twice: one pass leaves O(eps) components along Q
    return orthonormalize(G).float()


def overlap(Q1, Q2):
    """Mean squared cosine of the principal angles between two subspaces: |Q1^T Q2|_F^2 / min(r1, r2), 1 when one
    contains the other, r2 / W in expectation for random ones."""
    return ((Q1.T @ Q2).square().sum() / min(Q1.shape[1], Q2.shape[1])).item()


def capture(Q, G):
    """rho: the fraction of the energy of credit vectors G (n, W) inside span(Q)."""
    return ((G @ Q).square().sum() / G.square().sum().clamp_min(1e-30)).item()


def stream_sites(n_layer):
    return ['emb'] + [f'{kind}{i}' for i in range(n_layer) for kind in ('mid', 'out')]


def block_of(site, n_layer):
    """The block whose MLP gives a site its contribution statistics: its own block; the raw embedding takes block
    0's."""
    return 0 if site == 'emb' else int(site[3:])


class SubspaceBank:
    """Per-site orthonormal probe bases, their statistics, refreshes and exploration (see the module docstring)."""

    def __init__(self, kind, width, n_layer, rank, refresh=0, warmup=0, explore=0.0, explore_use='measure',
                 extra_rank=0, route_k=0, seed=0, device='cpu'):
        if kind not in KINDS:
            raise ValueError(f'Unknown basis kind {kind!r}; choose from {KINDS}.')
        if explore_use not in EXPLORE:
            raise ValueError(f'Unknown exploration use {explore_use!r}; choose from {EXPLORE}.')
        if kind != 'none' and not 0 < rank <= width:
            raise ValueError(f'Credit rank must be in 1..{width}, not {rank}.')
        self.kind, self.width, self.n_layer, self.rank = kind, width, n_layer, rank
        self.refresh_steps, self.warmup, self.explore, self.explore_use = refresh, warmup, explore, explore_use
        self.sentinels = 0 if kind == 'none' or not explore else max(1, round(explore * rank))
        if self.sentinels >= rank:
            raise ValueError('The exploration fraction leaves no in-subspace probes.')
        self.inside = rank - self.sentinels  # in-subspace probe directions per active site
        self.depth = min(width, max(rank, extra_rank))  # columns kept (adapters may take more than the probes)
        self.route_k, self.seed, self.device = route_k, seed, torch.device(device)
        self.sites = stream_sites(n_layer)
        self.Q, self.acquired = {}, {}
        self.generation, self.t = 0, 0
        self.metrics = {}  # this step's logged values (python floats or 0-d tensors)
        self._reset_statistics()
        self.sentinel_cov = {}
        if kind == 'random' and warmup == 0:
            self.refresh()

    # ------------------------------------------------------------------------------------------------ statistics

    def _reset_statistics(self):
        self.moment = {}   # stat key -> (W, W) float32 sum
        self.mean = {}     # stat key -> (W,) float32 sum (activation-pca only)
        self.count = {}    # stat key -> tokens

    def stat_key(self, site):
        if self.kind in ('contrib', 'route-cov'):
            return f'mlp{block_of(site, self.n_layer)}'
        return site

    def _add(self, key, moment, count, total=None):
        if key in self.moment:
            self.moment[key] += moment
            self.count[key] += count
            if total is not None:
                self.mean[key] += total
        else:
            self.moment[key], self.count[key] = moment.clone(), count
            if total is not None:
                self.mean[key] = total.clone()

    @property
    def needs_forward(self):
        return self.kind in ('activation-pca', 'contrib', 'route-cov') and self.collecting

    @property
    def collecting(self):
        """Whether statistics can still be used: not once the bases are built and never refresh."""
        return not (self.refresh_steps == 0 and self.Q and self.t >= self.warmup)

    @torch.no_grad()
    def observe_forward(self, streams, activations, writers, valid=None):
        """Forward statistics from one clean pass: streams {site: (n, W)}; activations [block] (n, H), the MLP's
        relu(hpre)^2; writers [block] (W, H), its c_proj weight; valid (n,) bool."""
        if not self.needs_forward:
            return
        if self.kind == 'activation-pca':
            for site, x in streams.items():
                x = x.float() if valid is None else x.float()[valid]
                self._add(site, x.T @ x, x.shape[0], x.sum(0))
            return
        for b, (a, w) in enumerate(zip(activations, writers)):
            a = a.float() if valid is None else a.float()[valid]
            w = w.float()
            norms = w.norm(dim=0)
            if self.kind == 'contrib':
                weights = a.sum(0) / norms.clamp_min(1e-12)  # |w_j|^2 a_j / |w_j|, the unit columns' weight
            else:
                top = (a * norms).topk(self.route_k or a.shape[1] // 16, dim=-1).indices
                weights = torch.zeros_like(a).scatter_(1, top, a.gather(1, top).square()).sum(0)
            self._add(f'mlp{b}', (w * weights) @ w.T, a.shape[0])

    @torch.no_grad()
    def observe_credit(self, site, g):
        """oracle-grad: the measured full-cycle credit (n, W) at a site."""
        if self.kind == 'oracle-grad' and self.collecting:
            g = g.float()
            self._add(site, g.T @ g, g.shape[0])

    # -------------------------------------------------------------------------------------------------- the bases

    def active(self, site):
        return self.kind != 'none' and site in self.Q

    def ready(self, sites):
        return all(self.active(s) for s in sites)

    def basis(self, site, rank=None):
        """The site's leading `rank` columns (W, rank), unit norm."""
        return self.Q[site][:, :rank or self.rank]

    def set_basis(self, site, Q):
        """Install a basis by hand (tests and diagnostics)."""
        self.Q[site] = Q.to(self.device, torch.float32)
        self.acquired[site] = self.t

    def directions(self, site):
        """This step's probe directions at an active site: (inside, W) in-subspace rows and (sentinels, W) rows in
        their complement, all unit norm."""
        rows = self.basis(site, self.inside)
        if not self.sentinels:
            return rows.T, rows.new_zeros(0, self.width)
        seed = self.seed * 1_000_003 + self.t * 9973 + self.sites.index(site) + 1
        return rows.T, complement_sample(rows, self.sentinels, seed).T

    def credit(self, site, derivative, rows, sentinel_rows):
        """The credit (n, W) from forward differences (n, inside + sentinels) along unit rows scaled to the Hadamard
        convention (|row|^2 = W): sum_j d_j q_j, i.e. derivative @ B / W with B = sqrt(W) Q^T. Logs the sentinels'
        estimate of the credit energy outside the subspace."""
        scale = math.sqrt(self.width)
        inside = derivative[:, :self.inside]
        credit = inside @ rows / scale
        if self.sentinels:
            m = self.sentinels
            outside = derivative[:, self.inside:self.inside + m]
            share = (self.width - self.inside) / m  # each sentinel samples m of the W - inside missing directions
            missing = outside.square().sum(1) * share / self.width
            kept = inside.square().sum(1) / self.width
            self.metrics[f'off_{site}'] = (missing.sum() / (missing.sum() + kept.sum()).clamp_min(1e-30))
            if self.explore_use == 'augment':
                part = outside @ sentinel_rows / scale
                self._add_sentinel(site, part.T @ part)
            elif self.explore_use == 'update':
                credit = credit + share * (outside @ sentinel_rows) / scale
        return credit

    def _add_sentinel(self, site, moment):
        self.sentinel_cov[site] = self.sentinel_cov.get(site, 0) + moment

    def project(self, site, g):
        """P g for an oracle-grad site (the full credit measured, then projected)."""
        Q = self.basis(site)
        return (g @ Q) @ Q.T

    # ------------------------------------------------------------------------------------------------- refreshing

    def end_step(self):
        """Advance one optimizer step; refresh the bases when due (at the end of warmup, then every refresh steps)."""
        self.t += 1
        if self.kind == 'none' or self.t < self.warmup:
            return
        since = self.t - self.warmup
        if since == 0 or (self.refresh_steps and since % self.refresh_steps == 0) or not self.Q:
            self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Rebuild every site's basis from the statistics gathered since the last refresh (sites without statistics
        keep their basis), then clear the statistics."""
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        old = dict(self.Q)
        new = {}
        if self.kind == 'random':
            for k, site in enumerate(self.sites):
                new[site] = random_basis(self.width, self.depth, self.seed * 7919 + 104729 * self.generation + k,
                                         self.device)
        else:
            for site in self.sites:
                key = self.stat_key(site)
                if key not in self.moment:
                    continue
                M, n = self.moment[key], torch.tensor(float(self.count[key]), device=self.device)
                if world > 1:
                    torch.distributed.all_reduce(M)
                    torch.distributed.all_reduce(n)
                M = M / n.clamp_min(1)
                if self.kind == 'activation-pca':
                    mu = self.mean[key]
                    if world > 1:
                        torch.distributed.all_reduce(mu)
                    mu = mu / n.clamp_min(1)
                    M = M - torch.outer(mu, mu)
                Q, _ = leading_eigenvectors(M, self.depth)
                if self.explore_use == 'augment' and site in self.sentinel_cov:
                    S = self.sentinel_cov[site]
                    if world > 1:
                        torch.distributed.all_reduce(S)
                    m = self.sentinels  # the found directions replace the weakest probed ones
                    found, _ = leading_eigenvectors(S, m)
                    keep = Q[:, :max(0, self.inside - m)]
                    found = found - keep @ (keep.T @ found)
                    Q = orthonormalize(torch.cat([keep, found, Q[:, keep.shape[1]:]], 1))[:, :self.depth]
                if world > 1:
                    if rank != 0:
                        Q = torch.empty_like(Q)
                    torch.distributed.broadcast(Q, 0)
                new[site] = Q
        if not new:
            return
        for site, Q in new.items():
            error = (Q.T @ Q - torch.eye(Q.shape[1], device=Q.device)).abs().max().item()
            if error > 1e-4:
                raise FloatingPointError(f'Basis at {site} is not orthonormal ({error:.2e}).')
            if site in old:
                self.metrics[f'turnover_{site}'] = 1 - overlap(old[site][:, :self.rank], Q[:, :self.rank])
            self.Q[site], self.acquired[site] = Q, self.t
        self.generation += 1
        self.metrics['generation'] = self.generation
        self._reset_statistics()
        self.sentinel_cov = {}

    # ----------------------------------------------------------------------------------------------- checkpoints

    def pop_metrics(self):
        metrics = {k: (v.item() if torch.is_tensor(v) else v) for k, v in self.metrics.items()}
        self.metrics = {}
        return metrics

    def state_dict(self):
        return dict(kind=self.kind, rank=self.rank, t=self.t, generation=self.generation,
                    Q={k: v.cpu() for k, v in self.Q.items()}, acquired=dict(self.acquired),
                    moment={k: v.cpu() for k, v in self.moment.items()}, mean={k: v.cpu() for k, v in self.mean.items()},
                    count=dict(self.count),
                    sentinel_cov={k: v.cpu() for k, v in self.sentinel_cov.items()})

    def load_state_dict(self, state):
        if (state['kind'], state['rank']) != (self.kind, self.rank):
            raise ValueError(f'Checkpoint basis is {state["kind"]} rank {state["rank"]}, not {self.kind} {self.rank}.')
        to = lambda d: {k: v.to(self.device) for k, v in d.items()}
        self.t, self.generation = state['t'], state['generation']
        self.Q, self.acquired = to(state['Q']), dict(state['acquired'])
        self.moment, self.mean, self.count = to(state['moment']), to(state['mean']), dict(state['count'])
        self.sentinel_cov = to(state['sentinel_cov'])
