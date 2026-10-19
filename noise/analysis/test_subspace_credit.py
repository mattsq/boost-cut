"""Correctness tests for circuit-guided subspace credit (docs/proposals/circuit-guided-low-rank-credit.md, section 9):
train/boostcut/subspace.py, projected_credit.py and adapters.py on a small model. Runs on CPU (compilation disabled;
the tests check the mathematics, not the kernels) or on a GPU.

    python -m pytest analysis/test_subspace_credit.py -q
"""
import copy
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
import torch._dynamo
from torch.nn import functional as F

torch._dynamo.config.disable = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train/boostcut'))
spec = importlib.util.spec_from_file_location('boostcut_cgsc_tests', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
spec = importlib.util.spec_from_file_location('test_boostcut_reference', ROOT / 'analysis/test_boostcut.py')
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
from adapters import UpdateLeakage, WriterUpdates, writer_names  # noqa: E402
from projected_credit import clean_gradients, make_estimators  # noqa: E402
import subspace  # noqa: E402
from subspace import SubspaceBank  # noqa: E402
from utils.model import norm  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WIDTH, LAYERS, VOCAB, T = 128, 2, 256, 24
bc.WIDTH, bc.HIDDEN = WIDTH, 4 * WIDTH
Boost, Cut = make_estimators(bc)
relative_error = lambda a, b: ((a - b).norm() / b.norm()).item()
cosine = lambda a, b: F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


@pytest.fixture(scope='module')
def model():
    config = bc.GPTConfig(n_layer=LAYERS, n_head=WIDTH // 64, n_kv_head=WIDTH // 64, n_embd=WIDTH, sequence_len=64,
                          vocab_size=VOCAB)
    model = bc.make_model(DEVICE, 0, config)
    with torch.no_grad():  # a little structure in the writers and head, so credits are not degenerate
        g = torch.Generator().manual_seed(1)
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.add_(0.02 * torch.randn(p.shape, generator=g).to(p))
    return model


@pytest.fixture(scope='module')
def batch():
    rows = torch.randint(0, VOCAB, (1, T + 1), generator=torch.Generator().manual_seed(2)).to(DEVICE)
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def settings():
    return bc.Settings(fc_k=bc.HIDDEN // 8, route_k=bc.HIDDEN // 16, local=dict(q=64, k=320, v=512, gate=64),
                       chain_tokens=8)


def bank(kind='random', rank=WIDTH // 4, **kwargs):
    b = SubspaceBank(kind, WIDTH, LAYERS, rank, route_k=bc.HIDDEN // 16, device=DEVICE, **kwargs)
    if kind == 'random' and not b.Q:
        b.refresh()
    return b


def hadamard_bank(est, sites):
    """A full-rank bank whose basis at each site is exactly the signed Hadamard cycle the full path draws there, in
    the order est.basis() is called; the estimator's generator is left untouched."""
    state = est.generator.get_state()
    b = SubspaceBank('random', WIDTH, LAYERS, WIDTH, device=DEVICE)
    if isinstance(est, bc.Cut):  # Cut.estimate probes the top stream (a basis and an offset) before the chain
        est.basis()
        torch.randint(0, WIDTH, (T, 1), device=DEVICE, generator=est.generator)
    for site in sites:
        b.set_basis(site, est.basis().T / math.sqrt(WIDTH))
        if not isinstance(est, bc.Cut):
            torch.randint(0, WIDTH, (T, 1), device=DEVICE, generator=est.generator)  # probe()'s offset draw
    est.generator.set_state(state)
    return b


# --------------------------------------------------------------------------------------- 1. projector mathematics

def test_projector_mathematics():
    A = torch.randn(WIDTH, WIDTH, device=DEVICE)
    for Q in (subspace.random_basis(WIDTH, 16, 0, DEVICE), subspace.leading_eigenvectors(A @ A.T, 16)[0]):
        P = Q @ Q.T
        assert torch.allclose(Q.T @ Q, torch.eye(16, device=DEVICE), atol=1e-5)
        assert torch.allclose(P @ P, P, atol=1e-5)
        C = subspace.complement_sample(Q, 8, 3)
        assert (Q.T @ C).abs().max() < 1e-5
    # a linear loss: exact directional derivatives along the probes give P g, nothing more
    b = bank(rank=16)
    g = torch.randn(5, WIDTH, device=DEVICE)
    rows, sentinels = b.directions('mid0')
    derivative = g @ (rows.T * math.sqrt(WIDTH))
    Q = b.basis('mid0')
    assert torch.allclose(b.credit('mid0', derivative, rows, sentinels), g @ Q @ Q.T, atol=1e-5)


def test_sentinels_measure_the_missing_energy():
    """With the sentinels spanning the whole complement, their estimate of the missing energy is exact, and 'update'
    recovers the full credit."""
    for use in ('measure', 'update'):
        b = bank(rank=WIDTH, explore=0.5, explore_use=use)
        g = torch.randn(7, WIDTH, device=DEVICE)
        rows, sentinels = b.directions('emb')
        assert sentinels.shape[0] == WIDTH - rows.shape[0]
        derivative = g @ (torch.cat([rows, sentinels]).T * math.sqrt(WIDTH))
        credit = b.credit('emb', derivative, rows, sentinels)
        Q = rows.T
        missing = (g - g @ Q @ Q.T).square().sum() / g.square().sum()
        assert abs(b.metrics['off_emb'].item() - missing.item()) < 1e-4
        expected = g if use == 'update' else g @ Q @ Q.T
        assert torch.allclose(credit, expected, atol=1e-4)


def test_augment_moves_found_directions_into_the_probed_columns():
    """Credit concentrated off the basis is found by the sentinels and, at the next refresh, probed. (Width 16: 4
    sentinels sample a 12-dimensional complement; at full width the same search needs proportionally more steps.)"""
    w = 16
    b = SubspaceBank('activation-pca', w, LAYERS, 8, warmup=1, refresh=60, explore=0.5, explore_use='augment',
                     device=DEVICE)
    g = torch.Generator().manual_seed(8)
    hidden = None
    for step in range(61):
        b.observe_forward({'mid0': torch.randn(64, w, generator=g).to(DEVICE)}, [], [])
        if b.active('mid0'):
            rows, sentinels = b.directions('mid0')
            if hidden is None:  # a credit direction outside the first basis
                hidden = subspace.complement_sample(b.basis('mid0'), 1, 5)[:, 0]
            credit = torch.randn(64, 1, generator=g).to(DEVICE) * hidden
            derivative = credit @ (torch.cat([rows, sentinels]).T * math.sqrt(w))
            b.credit('mid0', derivative, rows, sentinels)
        b.end_step()
    assert b.generation == 2
    probed = b.basis('mid0', b.inside)
    assert (probed.T @ hidden).square().sum().item() > 0.9


def test_refresh_without_statistics_is_not_a_generation():
    b = bank('oracle-grad', rank=8, warmup=1, refresh=1)
    for _ in range(3):
        b.end_step()
    assert b.generation == 0 and not b.Q

def test_bases_from_statistics():
    """route-cov is W diag(sum_t a_tj^2 [j in top_t]) W^T; the oracle recovers a planted credit subspace; a random
    basis captures about r / W of isotropic credit."""
    b = bank('route-cov', rank=8, warmup=1)
    a = F.relu(torch.randn(40, 4 * WIDTH, device=DEVICE)).square()
    w = torch.randn(WIDTH, 4 * WIDTH, device=DEVICE)
    b.observe_forward({}, [a] * LAYERS, [w] * LAYERS)
    top = (a * w.norm(dim=0)).topk(bc.HIDDEN // 16, dim=-1).indices
    v = a.gather(1, top).unsqueeze(-1) * w.T[top]  # (n, k, W): each token's route contributions
    M = torch.einsum('nkw,nkv->wv', v, v) / 40
    b.end_step()
    expected, _ = subspace.leading_eigenvectors(M, 8)
    assert subspace.overlap(b.basis('mid0'), expected) > 0.999
    planted = subspace.random_basis(WIDTH, 6, 9, DEVICE)
    oracle = bank('oracle-grad', rank=6, warmup=1)
    oracle.observe_credit('mid1', torch.randn(200, 6, device=DEVICE) @ planted.T)
    oracle.end_step()
    assert subspace.overlap(oracle.basis('mid1'), planted) > 0.999
    iso = torch.randn(4000, WIDTH, device=DEVICE)
    assert abs(subspace.capture(bank(rank=32).basis('mid0'), iso) - 32 / WIDTH) < 0.03


# ---------------------------------------------------------------------------------- 2. Hadamard compatibility

def test_full_rank_hadamard_basis_matches_the_full_cycle(model, batch):
    x, y = batch
    yf = y.flatten()
    stream = (torch.randn(T, WIDTH, generator=torch.Generator().manual_seed(4)) * 3).to(DEVICE)
    est = Boost(model, settings(), 0, 0, 1, bank=bank(), enabled=True)
    reference = bc.Boost(model, settings(), 0, 0, 1)
    est.bank = hadamard_bank(est, ['mid0'])
    est.queue = ['mid0']
    measured = est.probe(stream, yf)
    full = reference.probe(stream, yf)
    assert relative_error(measured, full) < 1e-3
    est.close(), reference.close()


def test_full_rank_hadamard_basis_matches_cut_chain(model, batch):
    x, y = batch
    est = Cut(model, settings(), 0, 0, 1, bank=bank(), enabled=True)
    est.bank = hadamard_bank(est, ['emb'] + [f'mid{i}' for i in range(LAYERS)])
    est.bank.set_basis(f'out{LAYERS - 1}', torch.eye(WIDTH, device=DEVICE))  # the top probe: any full basis
    est.estimate(x, y)
    reference = bc.Cut(model, settings(), 0, 0, 1)
    reference.estimate(x, y)
    for site in ['emb'] + list(range(LAYERS)):
        a, b = est.measured['chain'][site], reference.measured['chain'][site]
        assert relative_error(a, b) < 1e-3, site  # rounding: sqrt(W)-scaled rows through bfloat16 products
    for i in range(LAYERS):  # the mixing scalars' central differences ride along unchanged
        assert relative_error(est.measured['chain']['lam', i], reference.measured['chain']['lam', i]) < 1e-3
    est.close(), reference.close()


# ------------------------------------------------------------------------------------------- 3. normalization

def test_projected_readout_credit_is_P_g_not_scaled(model, batch):
    x, y = batch
    yf = y.flatten()
    stream = (torch.randn(T, WIDTH, generator=torch.Generator().manual_seed(5)) * 3).to(DEVICE)
    r = WIDTH // 4
    est = Boost(model, settings(), 0, 0, 1, bank=bank(rank=r), enabled=True)
    est.queue = ['mid0']
    measured = est.probe(stream, yf)
    weight = model.lm_head.weight.float()[:VOCAB]
    with torch.enable_grad():
        leaf = stream.clone().requires_grad_(True)
        bc.readout_losses(leaf, weight, yf).sum().backward()
    Q = est.bank.basis('mid0')
    projected = leaf.grad @ Q @ Q.T
    assert cosine(measured, projected) > 0.995
    ratio = (measured.norm() / projected.norm()).item()
    assert abs(ratio - 1) < 0.05, ratio  # (W / r) P g would be 4x
    est.close()


# --------------------------------------------------------------------- 4. surrogate autograd, 5. token isolation

def test_projected_cut_credit_is_P_times_surrogate_gradient(model, batch):
    x, y = batch
    s = settings()
    est = Cut(model, s, 0, 0, 1, bank=bank(rank=WIDTH // 4), enabled=True)
    est.estimate(x, y)
    reference = tb.cut_reference(model, x, y, s.route_k)
    for site, name, ref in [('emb', 'emb', reference['emb'])] + [(i, f'mid{i}', reference['mids'][i])
                                                                 for i in range(LAYERS)]:
        Q = est.bank.basis(name)
        projected = ref.float() @ Q @ Q.T
        measured = est.measured['chain'][site]
        assert cosine(measured, projected) > 0.98, (name, cosine(measured, projected))
        assert abs((measured.norm() / projected.norm()).item() - 1) < 0.1
    Q = est.bank.basis(f'out{LAYERS - 1}')
    assert cosine(est.measured['g_top'], reference['top'].float() @ Q @ Q.T) > 0.98
    est.close()


def test_each_token_is_its_own_experiment(model, batch):
    """The chain credit of a token does not depend on which other tokens share its chunk (attention stays frozen)."""
    x, y = batch
    credits = []
    for chunk in (1, T):
        s = settings()
        s.chain_tokens = chunk
        est = Cut(model, s, 0, 0, 1, bank=bank(rank=WIDTH // 4), enabled=True)
        est.estimate(x, y)
        credits.append(est.measured['chain'])
        est.close()
    for site in ['emb'] + list(range(LAYERS)):
        assert torch.allclose(credits[0][site], credits[1][site], rtol=1e-4, atol=1e-6), site


# -------------------------------------------------------------------------------------------- 6. writer gradient

def test_writer_gradient_of_the_left_factor():
    torch.manual_seed(6)
    Q = subspace.random_basis(WIDTH, 8, 1, DEVICE)
    P = Q @ Q.T
    ranks = []
    for _ in range(2):  # two ranks' minibatches, summed as the all-reduce does
        g, x = torch.randn(50, WIDTH, device=DEVICE), torch.randn(50, 3 * WIDTH, device=DEVICE)
        ranks.append(((g @ P).T @ x, g.T @ x))
    G_hat = sum(r[0] for r in ranks)
    G = sum(r[1] for r in ranks)
    G_A = Q.T @ G_hat
    assert torch.allclose(G_A, Q.T @ G, atol=1e-3)
    assert torch.allclose(Q @ G_A, P @ G, atol=1e-3)
    assert torch.linalg.matrix_rank(G_hat).item() <= 8


def writer_setup(model, mode, rank=8, merge=0):
    model = copy.deepcopy(model)
    b = bank(rank=rank)
    est = Boost(model, settings(), 0, 0, 1, bank=b, enabled=True)
    sites = est.writer_sites()
    est.close()
    writers = WriterUpdates(model, b, mode, rank, sites, lr=1e-3, merge_steps=merge)
    return model, b, sites, writers


def fake_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        p.grad = torch.randn(p.shape, generator=g).to(p)


def test_restricted_updates_stay_in_the_subspace(model):
    for mode in ('projected-sgd', 'qa', 'qa-merge'):
        m, b, sites, writers = writer_setup(model, mode, merge=2)
        named = dict(m.named_parameters())
        before = {n: named[n].detach().clone() for n in writers.params}
        for step in range(3):
            fake_grads(m, step)
            writers.step()
        for n in writers.params:
            delta = named[n] - before[n]
            Q = b.basis(sites[n], 8)
            assert (delta - Q @ (Q.T @ delta)).norm() < 1e-4 * max(1.0, delta.norm().item()), (mode, n)
    m, b, sites, writers = writer_setup(model, 'projected-sgd')
    fake_grads(m, 0)
    name = writer_names(LAYERS)[0]
    p = dict(m.named_parameters())[name]
    before, G = p.detach().clone(), p.grad.clone()
    writers.step()
    Q = b.basis(sites[name], 8)
    assert torch.allclose(p - before, -1e-3 * Q @ (Q.T @ G), atol=1e-6)


def test_leakage_of_dense_and_projected_steps(model):
    m = copy.deepcopy(model)
    b = bank(rank=8)
    sites = {n: 'mid0' for n in writer_names(LAYERS)}
    sgd = torch.optim.SGD([p for n, p in m.named_parameters() if n in sites], lr=1e-2)
    leak = UpdateLeakage(sgd, m, b, sites, every=1)
    fake_grads(m, 3)
    leak.step()
    assert leak.metrics['update_leakage'] > 0.9  # isotropic gradients: almost all of the step is outside rank 8
    for n, p in m.named_parameters():
        Q = b.basis('mid0')
        p.grad = Q @ (Q.T @ p.grad) if n in sites else p.grad
    leak.step()
    assert leak.metrics['update_leakage'] < 1e-4


# ----------------------------------------------------------------------------- 7. merged adapter, checkpoints

def test_merge_is_invariant_and_resume_is_exact(model):
    for mode in ('qa-merge', 'lora'):
        m, b, sites, writers = writer_setup(model, mode, merge=0)
        named = dict(m.named_parameters())
        for step in range(3):
            fake_grads(m, step)
            writers.step()
        values = {n: named[n].detach().clone() for n in writers.params}
        writers.merge()
        for n in writers.params:
            assert torch.equal(named[n], values[n])
            assert torch.allclose(writers.base[n], values[n])
        state, bank_state = copy.deepcopy(writers.state_dict()), copy.deepcopy(b.state_dict())
        weights = {n: p.detach().clone() for n, p in named.items()}

        def continue_(steps=3):
            for step in range(steps):
                fake_grads(m, 10 + step)
                writers.step()
            return {n: named[n].detach().clone() for n in writers.params}

        first = continue_()
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(weights[n])
        writers.load_state_dict(state)
        b.load_state_dict(bank_state)
        second = continue_()
        for n in first:
            assert torch.equal(first[n], second[n]), (mode, n)


def test_bank_checkpoint_round_trip():
    b = bank('activation-pca', rank=8, warmup=2, refresh=2)
    for step in range(3):
        b.observe_forward({'mid0': torch.randn(30, WIDTH, device=DEVICE)}, [], [])
        b.end_step()
    clone = SubspaceBank('activation-pca', WIDTH, LAYERS, 8, warmup=2, refresh=2, device=DEVICE)
    clone.load_state_dict(copy.deepcopy(b.state_dict()))
    data = torch.randn(30, WIDTH, device=DEVICE)
    for x in (b, clone):
        x.observe_forward({'mid0': data}, [], [])
        x.end_step()
    assert torch.equal(b.basis('mid0'), clone.basis('mid0')) and b.generation == clone.generation == 2


# -------------------------------------------------------------------------------------------- 8. no autograd

def test_forward_only_estimates(model, batch):
    x, y = batch
    for E in (Boost, Cut):
        est = E(model, settings(), 0, 0, 1, bank=bank(), enabled=True)
        clean_gradients(est, x, y)
        est.close()

    class Leaky:
        def estimate(self, a, b):
            torch.ones(1, requires_grad=True).sum().backward()

    with pytest.raises(RuntimeError, match='autograd called'):
        clean_gradients(Leaky(), x, y)


# ------------------------------------------------------------------------------------------------ 9. regression

@pytest.mark.parametrize('enabled_bank', [('none', True), ('random', False)])
def test_disabled_subspace_reproduces_the_original_estimates(model, batch, enabled_bank):
    kind, enabled = enabled_bank
    x, y = batch
    for E, Original in ((Boost, bc.Boost), (Cut, bc.Cut)):
        b = SubspaceBank(kind, WIDTH, LAYERS, 16, device=DEVICE)
        if kind == 'random':
            b.refresh()
        est = E(model, settings(), 0, 0, 1, bank=b, enabled=enabled)
        original = Original(model, settings(), 0, 0, 1)
        for _ in range(2):  # the second estimate checks that the random streams stay in step
            ours, theirs = est.estimate(x, y), original.estimate(x, y)
            for name in theirs:
                assert torch.equal(ours[name], theirs[name]), (E.__name__, name)
        est.close(), original.close()


# ------------------------------------------------------------------------------------------------ 10. accounting

def test_probe_counts_and_refresh_happen_inside_the_step(model, batch):
    x, y = batch
    m = copy.deepcopy(model)
    r = 16
    b = SubspaceBank('route-cov', WIDTH, LAYERS, r, warmup=1, refresh=1, explore=0.25, route_k=bc.HIDDEN // 16,
                     device=DEVICE)
    est = Cut(m, settings(), 0, 0, 1, bank=b, enabled=True)
    shapes = []
    original = bc.chain_chunk
    bc.chain_chunk = lambda site, move, *rest: shapes.append((site, move.shape[1])) or original(site, move, *rest)
    try:
        optimizer = bc.make_optimizer(m, bc.Recipe(), 10)
        est.step(x, y, optimizer)  # warmup: full cycles, statistics gathered, the first refresh at its end
        assert b.generation == 1 and all(P == WIDTH + 4 * (site >= 0) for site, P in shapes)
        shapes.clear()
        est.step(x, y, optimizer)
        assert all(P == r + 4 * (site >= 0) for site, P in shapes)  # r probes (sentinels included) + the scalars
        assert b.generation == 2
    finally:
        bc.chain_chunk = original
        est.close()
