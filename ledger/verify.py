"""Checks every number the paper prints against the run records, and the records against each other.

Each claim names where it is printed, the text the paper prints, and how to recompute it from the records: run
results (results/<run>/result.json), metrics (results/<run>/metrics.jsonl), the Modal logs saved under ledger/logs/,
and the independent re-scores (ledger/rescore.jsonl). A claim passes when the recomputed text equals the printed text
and that text appears in llm-slop-review/draft.md. Claims whose runs have not finished are reported as pending, never guessed.

Record checks: every re-scored checkpoint reproduces its logged validation and test loss; the official scorer agrees
on the benchmark model; every run scored the pinned validation split and the same test split.

    python3 ledger/verify.py            # writes ledger/verification.md, exits 1 on any failure
"""
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DRAFT_PATH = next(p for p in (ROOT / 'paper' / 'draft.md', ROOT / 'llm-slop-review' / 'draft.md') if p.exists())
DRAFT = DRAFT_PATH.read_text()  # the repository publishes paper/ as llm-slop-review/
IN_RELEASE = (ROOT / 'entries').exists()  # run from the published repository rather than the project
README_PATH = ROOT / 'README.md' if IN_RELEASE else ROOT / 'release' / 'README.md'
README = README_PATH.read_text() if README_PATH.exists() else ''
ENTRY_LOGS = {'track1_boostcut': ('submission/boostcut_track1_log', 'bc38f_s0'),
              'track2_boost': ('submission/boost_track2_log', 'boost_official_s0')}  # entry: (project log, its run)
VAL_SHA = '5fdb9242ce8ab83e519ab5dd5cf8d64f4d8a99f6e74c137f4ca0f723da5f7437'  # the benchmark's pinned val.pt
TEST_SHA = 'cb7a29f079ae2de5e96dd472497b67a6ec3c827bf6efbd8c263c7ab1001f6d7d'  # test.pt, as the Dust repository's


class Pending(Exception):
    pass


# ------------------------------------------------------------------------------------------------------ the records

def result(run):
    path = ROOT / 'results' / run / 'result.json'
    if not path.exists():
        raise Pending(run)
    return json.loads(path.read_text())


def field(runs, key):
    return [result(r)[key] for r in runs]


def log_lines(job):
    """JSON lines of a job's output: the Modal log if saved, else the local copy."""
    for path in (HERE / 'logs' / f'{job}.log', ROOT / 'localruns' / f'{job}.out'):
        if path.exists():
            out = []
            for line in path.read_text(errors='ignore').splitlines():
                line = line.strip()
                if line.startswith('{'):
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return out
    raise Pending(job)


def oracle(job, run):
    """Validation loss of one oracle schedule (analysis/oracle_hybrid.py)."""
    for d in log_lines(job):
        if d.get('run') == run and 'val_loss' in d:
            return d['val_loss']
    raise Pending(f'{job}:{run}')


def line_with(job, key):
    """The first JSON line of a job's output that has this key."""
    for d in log_lines(job):
        if key in d:
            return d
    raise Pending(f'{job}:{key}')


def keyed(job, key):
    return line_with(job, key)[key]


def val_at(run, seqs):
    for line in (ROOT / 'results' / run / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d.get('seqs') == seqs and 'val_loss' in d:
            return d['val_loss']
    raise Pending(f'{run}@{seqs}')


def rescores():
    path = HERE / 'rescore.jsonl'
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d['name']] = d
    return out


# ------------------------------------------------------------------------------------- Welch's t-test, no scipy needed

def _betacf(a, b, x, tiny=1e-300):
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_sf(t, dof):
    """P(T > t) for Student's t with dof degrees of freedom, t >= 0."""
    return 0.5 * _betainc(dof / 2, 0.5, dof / (dof + t * t))


def welch(xs, ys):
    """Difference of means, its 95% confidence interval, t, degrees of freedom and two-sided p (unequal variances)."""
    va, vb = statistics.variance(xs) / len(xs), statistics.variance(ys) / len(ys)
    diff, se = mean(xs) - mean(ys), math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(xs) - 1) + vb ** 2 / (len(ys) - 1))
    lo, hi = 0.0, 100.0
    for _ in range(200):  # the 97.5% quantile by bisection
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, dof) > 0.025 else (lo, mid)
    q = (lo + hi) / 2
    return diff, (diff - q * se, diff + q * se), diff / se, dof, 2 * t_sf(abs(diff / se), dof)


def pvalue(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


# ------------------------------------------------------------------------------------------------------- formatting

mean = statistics.mean
f3 = '{:.3f}'.format
f4 = '{:.4f}'.format


def pm(xs, digits=3):
    return f'{mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}'


def secs(xs):
    return f'{mean(xs):,.0f} s'


def times(x, digits=0):
    return f'{x:,.{digits}f}×'


# ------------------------------------------------------------------------------------------------------- the claims

S = lambda *names: list(names)
BC = S('bc38f_s0', 'bc38f_s1', 'bc38f_s2')            # the final configuration
BASE = S('bc38_s0', 'bc38_s1', 'bc38_s2')             # Boost's keys sampled: the ablations' base
BP = S('bp38_s0', 'bp38_s1', 'bp38_s2')
DUST = S('dust38_s0', 'dust38_s1', 'dust38_s2')
CUT = S('bc38_sw0', 'bc38_sw0_s1', 'bc38_sw0_s2')
BOOST = S('bc38f_sw100_s0', 'bc38f_sw100_s1', 'bc38f_sw100_s2')


def flops(job, method):
    return keyed(job, method)['flops_per_token']


def boostcut_flops(job):
    """Per token over the epoch: Boost on 30% of the tokens, Cut on 70%."""
    return 0.3 * flops(job, 'boost') + 0.7 * flops(job, 'cut')


def gap(bc_runs, bp_runs):
    return mean(field(bc_runs, 'test_loss')) - mean(field(bp_runs, 'test_loss'))


def mech():
    """analysis/clean_probe.py's records per checkpoint, keyed bc_<seqs> (Boost -> Cut seed 1) and cut_<seqs>."""
    out = {}
    for d in log_lines('mech38'):
        out.update({k: v for k, v in d.items() if k.startswith(('bc_', 'cut_'))})
    return out


def fidelity(checkpoint, k):
    return min(mech()[checkpoint]['routes'][str(k)]['token_cos'].values())


def k90(checkpoint, blocks):
    return [mech()[checkpoint]['sparsity'][b]['k90'] for b in blocks]


def span(values, fmt='{:,.0f}'.format):
    return f'{fmt(min(values))}–{fmt(max(values))}'


# with and without the head start, nearest in validation loss (checkpoints every 488 sequences)
PAIRS = [(1464, 1464), (2440, 2928), (2928, 3416), (4392, 4392)]


def mech_row(pair):
    a, b = pair
    with_, without = f'bc_{a}', f'cut_{b}'
    return (f'{f3(val_at("bc38_s1", a))} / {f3(val_at("bc38_sw0", b))}',
            ' / '.join(f3(fidelity(with_, k)) for k in (64, 128, 256)),
            ' / '.join(f3(fidelity(without, k)) for k in (64, 128, 256)),
            f'{span(k90(with_, range(1, 8)))} / {span(k90(without, range(1, 8)))}')


def cosines(values):
    lo, hi = f'{min(values):.4f}', f'{max(values):.4f}'
    return lo if lo == hi else f'{lo}–{hi}'


def autograd_row(label, job, k, k_all=None):
    """Appendix B: cosine with autograd at the production route budget k (chain sites: the raw embedding and every
    mid stream), at all units, for the top stream, the mixing scalars and the exact keys (Cut's and Boost's)."""
    d = keyed(job, f'cut_k{k}')
    full = cosines([keyed(job, f'cut_k{k_all}')['emb']] + keyed(job, f'cut_k{k_all}')['mids']) if k_all else '–'
    scalars = f'{d["lam"][0]:.3f} (ratio {d["lam"][1]:.3f})' if 'plain' not in label else '–'
    keys = [c for c, _ in d['keys'].values()] + [c for c, _ in keyed(job, 'boost_exact_keys')['keys'].values()]
    return (f'| {label} | {k} | {cosines([d["emb"]] + d["mids"])} | {full} | {d["top"]:.3f} | {scalars} | '
            f'{min(keys):.3f} |')


def after_switch(prefix):
    """Checkpoints from the switch on (1,464 sequences), in order."""
    return sorted((k for k in mech() if k.startswith(prefix + '_') and int(k.split('_')[1]) >= 1464),
                  key=lambda k: int(k.split('_')[1]))


CLAIMS = [
    # where, printed, recompute
    ('Abstract: Boost → Cut test, 3 seeds', '3.935 ± 0.012', lambda: pm(field(BC, 'test_loss'))),
    ('Abstract: Boost → Cut minutes', '7.3 minutes', lambda: f'{mean(field(BC, "train_seconds")) / 60:.1f} minutes'),
    ('Abstract: matrix FLOPs, Dust / Boost → Cut', '24× fewer', lambda: times(flops('flops38b', 'dust') / boostcut_flops('flops38b')) + ' fewer'),
    ('Abstract: route cost after / without the head start (oracle)', '0.005 nats of exact-credit loss after the head start and 0.028',
     lambda: f'{oracle("oh_route128", "attn_f0.3") - oracle("oh_base", "attn_f0.3"):.3f} nats of exact-credit loss after the head start and '
             f'{oracle("oh_route128", "attn_f0.0") - oracle("oh_base", "attn_f0.0"):.3f}'),
    ('Abstract: FLOPs vs backprop', '136×', lambda: times(boostcut_flops('flops38b') / flops('flops38b', 'backprop'))),
    ('Abstract: attention-cut share of the gap', '0.19 of its 0.23-nat gap',
     lambda: f'{oracle("oh_base", "attn_f0.3") - oracle("oracle_hybrid", "full_f0.3"):.2f} of its '
             f'{mean(field(BC, "val_loss")) - mean(field(BP, "val_loss")):.2f}-nat gap'),
    ('Contribution 2: route cost and speedup (all units vs top-128)', 'costs 0.006 nats of measured test loss and saves 2.4× the time',
     lambda: f'costs {mean(field(BASE, "test_loss")) - result("bc38_kall")["test_loss"]:.3f} nats of measured test loss and saves '
             f'{result("bc38_kall")["train_seconds"] / mean(field(BASE, "train_seconds")):.1f}× the time'),
    ('Section 3.5: backprop FLOPs per token', "backprop's 0.264 GFLOP", lambda: f"backprop's {flops('flops38b', 'backprop') / 1e9:.3f} GFLOP"),
    ('Section 3.5: Boost FLOPs', '| Boost | 29.0 G | 110× |',
     lambda: f"| Boost | {flops('flops38b', 'boost') / 1e9:.1f} G | {times(flops('flops38b', 'boost') / flops('flops38b', 'backprop'))} |"),
    ('Section 3.5: Cut FLOPs', '| Cut | 38.8 G | 147× |',
     lambda: f"| Cut | {flops('flops38b', 'cut') / 1e9:.1f} G | {times(flops('flops38b', 'cut') / flops('flops38b', 'backprop'))} |"),
    ('Section 3.5: Boost vocabulary entries per token', '(50M vocabulary entries per token)',
     lambda: f"({keyed('flops38b', 'boost')['vocab_evals_per_token'] / 1e6:.0f}M vocabulary entries per token)"),
    ('Section 3.5: Cut vocabulary entries per token', '(21M vocabulary entries per token)',
     lambda: f"({keyed('flops38b', 'cut')['vocab_evals_per_token'] / 1e6:.0f}M vocabulary entries per token)"),
    ('Section 3.5: Cut at 243M', 'Cut costs 227× backprop (371 against 1.64 GFLOP per token)',
     lambda: f"Cut costs {times(flops('flops243', 'cut') / flops('flops243', 'backprop'))} backprop "
             f"({flops('flops243', 'cut') / 1e9:.0f} against {flops('flops243', 'backprop') / 1e9:.2f} GFLOP per token)"),
    ('Section 3.5: Boost at 243M', '(54× at 243M)', lambda: f"({times(flops('flops243', 'boost') / flops('flops243', 'backprop'))} at 243M)"),
    ('Section 3.5: Boost → Cut at 38M and 243M', 'Boost → Cut is 136× at 38M\nand 175× at 243M',
     lambda: f"Boost → Cut is {times(boostcut_flops('flops38b') / flops('flops38b', 'backprop'))} at 38M\n"
             f"and {times(boostcut_flops('flops243') / flops('flops243', 'backprop'))} at 243M"),
    ('Table 1: Boost → Cut', f'| **Boost → Cut** | **3.935 ± 0.012** | 3.946 ± 0.013 | 438 s |',
     lambda: f'| **Boost → Cut** | **{pm(field(BC, "test_loss"))}** | {pm(field(BC, "val_loss"))} | {secs(field(BC, "train_seconds"))} |'),
    ('Table 1: Cut alone', None,
     lambda: f'| Cut alone (no head start) | {pm(field(CUT, "test_loss"))} | {pm(field(CUT, "val_loss"))} | {secs(field(CUT, "train_seconds"))} |'),
    ('Table 1: Boost alone', None, lambda: f'| Boost alone | {pm(field(BOOST, "test_loss"))} | {pm(field(BOOST, "val_loss"))} | {secs(field(BOOST, "train_seconds"))} |'),
    ('Table 1: Dust record', None,
     lambda: f'| Dust record (16k population) | {pm(field(DUST, "test_loss"))} | {pm(field(DUST, "val_loss"))} | {secs(field(DUST, "train_seconds"))} |'),
    ('Table 1: forward gradient', '| local losses + forward gradient (128 directions) | 4.292 | 4.305 | 474 s |',
     lambda: f'| local losses + forward gradient (128 directions) | {f3(result("bc38_fg128")["test_loss"])} | '
             f'{f3(result("bc38_fg128")["val_loss"])} | {secs([result("bc38_fg128")["train_seconds"]])} |'),
    ('Table 1: backprop', '| backprop, same model, init and optimizer | 3.706 ± 0.009 | 3.712 ± 0.009 | 31 s |',
     lambda: f'| backprop, same model, init and optimizer | {pm(field(BP, "test_loss"))} | {pm(field(BP, "val_loss"))} | {secs(field(BP, "train_seconds"))} |'),
    ('Section 4.1: official log', 'validation 3.9448, re-scored by `utils/eval.py` as\n3.94481, in 456 s',
     lambda: f'validation {f4(result("bc38f_s0")["val_loss"])}, re-scored by `utils/eval.py` as\n'
             f'{rescores()["bc38f_s0"]["official_val"]:.5f}, in {result("bc38f_s0")["train_seconds"]:.0f} s'),
    ('Table 2: base', '| **base: 30% switch, top-128 routes, exact Cut keys** (3 seeds) | **3.945 ± 0.003** | 3.956 ± 0.003 | 434 s | 3.910 |',
     lambda: f'| **base: 30% switch, top-128 routes, exact Cut keys** (3 seeds) | **{pm(field(BASE, "test_loss"))}** | '
             f'{pm(field(BASE, "val_loss"))} | {secs(field(BASE, "train_seconds"))} | {f3(oracle("oh_route128", "attn_f0.3"))} |'),
]


def single(label, run, oracle_value=None):
    def compute():
        r = result(run)
        row = f'| {label} | {f3(r["test_loss"])} | {f3(r["val_loss"])} | {secs([r["train_seconds"]])} |'
        return row + (f' {oracle_value()} |' if oracle_value else '')
    return compute


CLAIMS += [
    ('Table 2: switch 0%', '| switch 0% (Cut alone) | 3.979 | 3.992 | 503 s | 3.955 |',
     single('switch 0% (Cut alone)', 'bc38_sw0', lambda: f3(oracle('oh_route128', 'attn_f0.0')))),
    ('Table 2: switch 50%', '| switch 50% | 3.980 | 3.989 | 422 s | 3.938 (all units) |',
     single('switch 50%', 'bc38_sw50', lambda: f3(oracle('oracle_hybrid', 'attn_f0.5')) + ' (all units)')),
    ('Table 2: switch 100%', '| switch 100% (Boost alone) | 4.113 | 4.122 | 338 s | 4.060 |',
     single('switch 100% (Boost alone)', 'bc38_sw100', lambda: f3(oracle('oracle_hybrid', 'boost_f1.0')))),
    ('Table 2: routes 64', '| routes: top 64 (3% of units) | 3.967 | 3.976 | 434 s | 3.928 |',
     single('routes: top 64 (3% of units)', 'bc38_k64', lambda: f3(oracle('oh_route64', 'attn_f0.3')))),
    ('Table 2: routes 256', '| routes: top 256 (12.5%) | 3.935 | 3.945 | 489 s | |', lambda: single('routes: top 256 (12.5%)', 'bc38_k256')() + ' |'),
    ('Table 2: all units', '| routes: all 2,048 units | 3.939 | 3.948 | 1,021 s | 3.905 |',
     single('routes: all 2,048 units', 'bc38_kall', lambda: f3(oracle('oh_base', 'attn_f0.3')))),
    ('Table 2: Cut keys sampled', "| Cut's keys sampled (the record's search) | 3.969 | 3.980 | 464 s | |",
     lambda: single("Cut's keys sampled (the record's search)", 'bc38_ksampled')() + ' |'),
    ('Table 2: final configuration', "| Boost's keys exact too (the final configuration; 3 seeds) | 3.935 ± 0.012 | 3.946 ± 0.013 | 438 s | |",
     lambda: f"| Boost's keys exact too (the final configuration; 3 seeds) | {pm(field(BC, 'test_loss'))} | {pm(field(BC, 'val_loss'))} | "
             f"{secs(field(BC, 'train_seconds'))} | |"),
    ('Section 4.2: route speed lever', 'Top-128 costs about\n  0.006 for a 2.35× speedup',
     lambda: f'Top-128 costs about\n  {mean(field(BASE, "test_loss")) - result("bc38_kall")["test_loss"]:.3f} for a '
             f'{result("bc38_kall")["train_seconds"] / mean(field(BASE, "train_seconds")):.2f}× speedup'),
    ('Section 4.2: exact keys in Cut', 'Exact keys matter in Cut** (0.025)',
     lambda: f'Exact keys matter in Cut** ({result("bc38_ksampled")["test_loss"] - mean(field(BASE, "test_loss")):.3f})'),
    ('Section 4.3: forward gradient vs Boost alone', 'It\n  reaches 4.292 against 4.113',
     lambda: f'It\n  reaches {f3(result("bc38_fg128")["test_loss"])} against {f3(result("bc38_sw100")["test_loss"])}'),
    ('Section 4.3: DFA at the record rates', 'settles at test 5.439', lambda: f'settles at test {f3(result("bc38_dfa")["test_loss"])}'),
    ('Section 4.4: backprop oracle', '| backprop (true gradient) | 3.708 |', lambda: f'| backprop (true gradient) | {f3(oracle("oracle_hybrid", "full_f0.0"))} |'),
    ('Section 4.4: backprop oracle with head start', "| … with Boost's 30% head start | 3.711 |",
     lambda: f"| … with Boost's 30% head start | {f3(oracle('oracle_hybrid', 'full_f0.3'))} |"),
    ('Section 4.4: attention cut', '| exact attention-cut credit after the head start, all units | 3.905 | +0.195: the attention cut |',
     lambda: f'| exact attention-cut credit after the head start, all units | {f3(oracle("oh_base", "attn_f0.3"))} | '
             f'+{oracle("oh_base", "attn_f0.3") - oracle("oracle_hybrid", "full_f0.3"):.3f}: the attention cut |'),
    ('Section 4.4: routes', '| … top-128 routes | 3.910 | +0.005: the routes |',
     lambda: f'| … top-128 routes | {f3(oracle("oh_route128", "attn_f0.3"))} | '
             f'+{oracle("oh_route128", "attn_f0.3") - oracle("oh_base", "attn_f0.3"):.3f}: the routes |'),
    ('Section 4.4: measurement', '| measured: Boost → Cut (3 seeds) | 3.946 ± 0.013 | +0.036: measurement',
     lambda: f'| measured: Boost → Cut (3 seeds) | {pm(field(BC, "val_loss"))} | '
             f'+{mean(field(BC, "val_loss")) - oracle("oh_route128", "attn_f0.3"):.3f}: measurement'),
    ('Table 3: 7.3M', '| 7.3M (L4, d256) | 4.174 (115 s) | 3.915 (14 s) | 0.26 |',
     lambda: f'| 7.3M (L4, d256) | {f3(result("bc7_s0")["test_loss"])} ({result("bc7_s0")["train_seconds"]:.0f} s) | '
             f'{f3(result("bp7_s0")["test_loss"])} ({result("bp7_s0")["train_seconds"]:.0f} s) | {gap(["bc7_s0"], ["bp7_s0"]):.2f} |'),
    ('Table 3: 38M', '| 38M (L8, d512) | 3.935 ± 0.012 (438 s) | 3.706 ± 0.009 (31 s) | 0.23 |',
     lambda: f'| 38M (L8, d512) | {pm(field(BC, "test_loss"))} ({secs(field(BC, "train_seconds"))}) | '
             f'{pm(field(BP, "test_loss"))} ({secs(field(BP, "train_seconds"))}) | {gap(BC, BP):.2f} |'),
    ('Table 3: 243M backprop', '| 3.632 (67 s) |', lambda: f'| {f3(result("bp243_s0")["test_loss"])} ({result("bp243_s0")["train_seconds"]:.0f} s) |'),
    ("Table 3: 243M, Boost's keys sampled", "3.888 with Boost's keys sampled (3,209 s)",
     lambda: f"{f3(result('bc243_s0')['test_loss'])} with Boost's keys sampled ({result('bc243_s0')['train_seconds']:,.0f} s)"),
    ('Section 4.6: TinyStories Boost → Cut', '| Boost → Cut | 1.633 | 1.685 |',
     lambda: f'| Boost → Cut | {f3(result("bcts_s0")["test_loss"])} | {f3(result("bcts_s0")["val_loss"])} |'),
    ('Section 4.6: TinyStories backprop', '| backprop | 1.508 | 1.558 |',
     lambda: f'| backprop | {f3(result("bpts_s0")["test_loss"])} | {f3(result("bpts_s0")["val_loss"])} |'),
    ('Section 6: memory', '  - backprop: 0.80 GiB;\n  - Boost: 3.49 GiB;\n  - Cut: 3.57 GiB.',
     lambda: (lambda m: f"  - backprop: {m['backprop']['peak_gib']:.2f} GiB;\n  - Boost: {m['boost']['peak_gib']:.2f} GiB;\n"
                        f"  - Cut: {m['cut']['peak_gib']:.2f} GiB.")(line_with('memory38', 'preset'))),
    ('Section 6: memory ratio', 'about 4.4× backprop',
     lambda: (lambda m: f"about {(m['boost']['peak_gib'] + m['cut']['peak_gib']) / 2 / m['backprop']['peak_gib']:.1f}× backprop")(
         line_with('memory38', 'preset'))),
    ('Section 6: Dust FLOPs', 'The Dust record uses 876 GFLOP (3,316×)',
     lambda: f"The Dust record uses {flops('flops38b', 'dust') / 1e9:.0f} GFLOP ({times(flops('flops38b', 'dust') / flops('flops38b', 'backprop'))})"),
    ('Section 6: Boost → Cut FLOPs', 'Boost → Cut ≈ 35.8 GFLOP, 136× backprop',
     lambda: f"Boost → Cut ≈ {boostcut_flops('flops38b') / 1e9:.1f} GFLOP, {times(boostcut_flops('flops38b') / flops('flops38b', 'backprop'))} backprop"),
    ('Section 8: wall-clock vs backprop', 'about 14× its wall-clock',
     lambda: f"about {mean(field(BC, 'train_seconds')) / mean(field(BP, 'train_seconds')):.0f}× its wall-clock"),
    ('Appendix D: test split hash', TEST_SHA, lambda: rescores()['bc38f_s0']['split_token_sha256']['test']),
    ('Section 4.1: head start, Welch', 'Boost → Cut beats Cut alone by 0.049 (95% CI 0.023–0.074, p = 0.007)',
     lambda: (lambda d, ci, t, dof, p: f'Boost → Cut beats Cut alone by {-d:.3f} (95% CI {-ci[1]:.3f}–{-ci[0]:.3f}, '
                                         f'{pvalue(p)})')(*welch(field(BC, 'test_loss'), field(CUT, 'test_loss')))),
    ('Section 4.1: end-to-end phase, Welch', 'Boost → Cut beats Boost alone by 0.127 (95% CI 0.101–0.152, p < 0.001)',
     lambda: (lambda d, ci, t, dof, p: f'Boost → Cut beats Boost alone by {-d:.3f} (95% CI {-ci[1]:.3f}–{-ci[0]:.3f}, '
                                         f'{pvalue(p)})')(*welch(field(BC, 'test_loss'), field(BOOST, 'test_loss')))),
    ('Section 1: results in brief', '(by 0.049 and 0.127 nats; p = 0.007 and p < 0.001 over 3 seeds)',
     lambda: (lambda a, b: f'(by {-a[0]:.3f} and {-b[0]:.3f} nats; {pvalue(a[4])} and {pvalue(b[4])} over 3 seeds)')(
         welch(field(BC, 'test_loss'), field(CUT, 'test_loss')), welch(field(BC, 'test_loss'), field(BOOST, 'test_loss')))),
    ('Table 1: DFA, best rate', None,
     lambda: f'| direct feedback alignment (best of four learning rates) | {f3(result("bc38_dfa_lr0.1")["test_loss"])} | '
             f'{f3(result("bc38_dfa_lr0.1")["val_loss"])} | {secs([result("bc38_dfa_lr0.1")["train_seconds"]])} |'),
    ('Section 4.1: backprop learning rates', None,
     lambda: 'test ' + ', '.join(f3(result(r)['test_loss']) for r in ('bp38_lr0.5', 'bp38_s0', 'bp38_lr2'))
             + f' and {f3(result("bp38_lr4")["test_loss"])} at ×0.5, ×1, ×2 and ×4'),
    ('Section 4.1: Boost → Cut learning rates', None,
     lambda: ', '.join(f3(result(r)['test_loss']) for r in ('bc38f_lr0.5', 'bc38f_s0'))
             + f' and {f3(result("bc38f_lr2")["test_loss"])} at ×0.5, ×1 and ×2'),
    ('Section 4.1: baselines learning rates', None,
     lambda: f'forward gradient ×1 (×0.5: {f3(result("bc38_fg128_lr0.5")["test_loss"])}), and DFA ×0.1 '
             f'(×1: {f3(result("bc38_dfa")["test_loss"])},\n  ×0.3: {f3(result("bc38_dfa_lr0.3")["test_loss"])}, '
             f'×0.03: {f3(result("bc38_dfa_lr0.03")["test_loss"])})'),
    ('Section 4.3: DFA best', None, lambda: f'reaches only {f3(result("bc38_dfa_lr0.1")["test_loss"])}'),
    ('Section 4.6: plain Boost → Cut', None,
     lambda: f'| Boost → Cut, top-128 routes | {f3(result("plain_bc38_s0")["test_loss"])} | {f3(result("plain_bc38_s0")["val_loss"])} | '
             f'{secs([result("plain_bc38_s0")["train_seconds"]])} |'),
    ('Section 4.6: plain backprop', None,
     lambda: f'| backprop, same optimizer | {f3(result("plain_bp38_s0")["test_loss"])} | {f3(result("plain_bp38_s0")["val_loss"])} | '
             f'{secs([result("plain_bp38_s0")["train_seconds"]])} |'),
    ('Section 4.6 and 8: plain gap', None,
     lambda: f'gap to backprop grows from {gap(BC, BP):.2f} to {gap(["plain_bc38_s0"], ["plain_bp38_s0"]):.2f}'),
    ('Abstract: plain GPT without the shortcut', None,
     lambda: f'with all units, the gap to backprop is {gap(["plain_bc38_kall"], ["plain_bp38_s0"]):.2f} nats, at '
             f'{result("plain_bc38_kall")["train_seconds"] / result("plain_bc38_s0")["train_seconds"]:.1f}× the time'),
    ('Section 4.6: plain all-units row', None,
     lambda: f'| Boost → Cut, all 2,048 units | {f3(result("plain_bc38_kall")["test_loss"])} | '
             f'{f3(result("plain_bc38_kall")["val_loss"])} | {secs([result("plain_bc38_kall")["train_seconds"]])} |'),
    ('Section 4.6: all units, test', None,
     lambda: f'Boost → Cut reaches test {f3(result("plain_bc38_kall")["test_loss"])}, and Cut\nimproves at every evaluation'
     if (lambda v: all(later < earlier for earlier, later in zip(v, v[1:])))(
         [val_at('plain_bc38_kall', q) for q in (1952, 2440, 2928, 3416, 3904, 4392)] + [result('plain_bc38_kall')['val_loss']])
     else 'Cut does not improve at every evaluation'),
    ('Section 4.6: all units, gap and time', None,
     lambda: f'The gap to backprop is {gap(["plain_bc38_kall"], ["plain_bp38_s0"]):.2f}, against {gap(BC, BP):.2f} on the '
             f'benchmark model, at {result("plain_bc38_kall")["train_seconds"] / result("plain_bc38_s0")["train_seconds"]:.1f}× '
             f'the time of\nthe routed run'),
    ('Section 8: plain GPT', None,
     lambda: f'it fails (gap {gap(["plain_bc38_s0"], ["plain_bp38_s0"]):.2f}), and\n  without it the gap is '
             f'{gap(["plain_bc38_kall"], ["plain_bp38_s0"]):.2f} at '
             f'{result("plain_bc38_kall")["train_seconds"] / result("plain_bc38_s0")["train_seconds"]:.1f}× the time'),
    ('Section 4.6: Cut phase curves', None,
     lambda: 'validation goes ' + ', '.join(f'{val_at("plain_bc38_s0", q):.2f}' for q in (1952, 2440, 2928, 3416, 3904))
             + f' and {val_at("plain_bc38_s0", 4392):.2f} over its six evaluations, where the\n  benchmark model falls '
             f'steadily from {val_at("bc38f_s0", 1952):.2f} to {val_at("bc38f_s0", 4392):.2f}'),
    ('Section 4.2: exact keys in Boost alone', 'In Boost alone they matter as much (4.062 ± 0.007 against 4.113',
     lambda: f'In Boost alone they matter as much ({pm(field(BOOST, "test_loss"))} against {f3(result("bc38_sw100")["test_loss"])}'),
]

def entry_log(name):
    """An entry's official result.json, as submitted."""
    path = ROOT / 'entries' / name / 'log' / 'result.json' if IN_RELEASE else ROOT / ENTRY_LOGS[name][0] / 'result.json'
    return json.loads(path.read_text())


def record_log(track):
    """The benchmark record's own official log (train/val or train/speedrun)."""
    return json.loads((ROOT / 'noise' / 'train' / track / 'log' / 'result.json').read_text())


def entry_row(label, track, entry, record, bold_time):
    e, r = entry_log(entry), record_log(record)
    val = f'**{e["val_loss"]:.3f}**' if not bold_time else f'{e["val_loss"]:.3f}'
    time_ = f'**{e["train_seconds"]:.0f} s**' if bold_time else f'{e["train_seconds"]:.0f} s'
    return f'| **{label}** | {track} | {val} | {time_} | {r["val_loss"]:.3f} in {r["train_seconds"]:,.0f} s |'


def fidelity_log(job):
    """analysis/route_fidelity.py's records: {seqs: record}."""
    return {d['seqs']: d for d in log_lines(job) if 'fidelity' in d}


def after(job, seqs=1464):
    return [d for q, d in sorted(fidelity_log(job).items()) if q >= seqs]


CLAIMS += [
    ('Section 4.6: benchmark route fidelity', None,
     lambda: f"cosine {min(d['fidelity']['128'] for d in after('fid_noise')):.2f}–"
             f"{max(d['fidelity']['128'] for d in after('fid_noise')):.2f} at every checkpoint from the switch"),
    ('Section 4.6: benchmark units carrying 90%', None,
     lambda: f"at most {max(max(d['k90']) for d in after('fid_noise')):,} of 2,048 units carry 90%"),
    ('Section 4.6: plain route fidelity', None,
     lambda: f"carry it at {fidelity_log('fid_plain')[1464]['fidelity']['128']:.2f} at the switch, falling to "
             f"{fidelity_log('fid_plain')[4880]['fidelity']['128']:.2f} by the end"),
    ('Section 4.6: plain top 512', None, lambda: f"reach only {fidelity_log('fid_plain')[4880]['fidelity']['512']:.2f}"),
    ('Section 4.6: plain units carrying 90%', None,
     lambda: f"grow from {min(fidelity_log('fid_plain')[1464]['k90'])}–{max(fidelity_log('fid_plain')[1464]['k90'])} at the "
             f"switch to as many as {max(fidelity_log('fid_plain')[4880]['k90']):,}"),
]


README_CLAIMS = [
    ('README: Track 1 row', None, lambda: entry_row('Boost → Cut', '1: lowest loss', 'track1_boostcut', 'val', False)),
    ('README: Track 2 row', None, lambda: entry_row('Boost', '2: time to 4.2', 'track2_boost', 'speedrun', True)),
    ('README: speedrun target reached', None,
     lambda: 'time to 4.2' if entry_log('track2_boost')['reached_target'] and entry_log('track2_boost')['val_loss'] <= 4.2
     else 'target missed'),
    ('README: Boost → Cut over 3 seeds', None, lambda: pm(field(BC, 'test_loss'))),
    ('README: backprop over 3 seeds', None, lambda: pm(field(BP, 'test_loss'))),
    ('README: same hardware', None,
     lambda: f"Track 1 record's recipe reproduces at {f3(result('dust38_s0')['val_loss'])} in "
             f"{result('dust38_s0')['train_seconds']:,.0f} s, and the speedrun record's script runs in "
             f"{result('dust_speedrun_official2')['train_seconds']:,.0f} s, so Boost\nis "
             f"{result('dust_speedrun_official2')['train_seconds'] / entry_log('track2_boost')['train_seconds']:.1f}× faster"),
    ('README: why Cut, Boost alone stalls', None,
     lambda: f'Boost alone stalls at {mean(field(BOOST, "test_loss")):.3f} test loss'),
    ('README: why Cut, routes', None,
     lambda: f'makes it {result("bc38_kall")["train_seconds"] / mean(field(BASE, "train_seconds")):.1f}× cheaper, for '
             f'{mean(field(BASE, "test_loss")) - result("bc38_kall")["test_loss"]:.3f} of loss'),
    ('README: why Cut, head start', None,
     lambda: f"With Boost's head start Cut reaches {mean(field(BC, 'test_loss')):.3f}; without it, "
             f"{mean(field(CUT, 'test_loss')):.3f}"),
]


for i, pair in enumerate(PAIRS, 1):
    CLAIMS.append((f'Table 4: row {i}', None, (lambda p: lambda: '| ' + ' | '.join(mech_row(p)) + ' |')(pair)))

CLAIMS += [
    ('Section 5: block 0 with / without the head start', '7–22 units carry 90% of its output, against 91–274',
     lambda: f"{span([mech()[c]['sparsity'][0]['k90'] for c in after_switch('bc')])} units carry 90% of its output, against "
             f"{span([mech()[c]['sparsity'][0]['k90'] for c in after_switch('cut')])}"),
    ('Section 5: saliency coordinate peak', '(peak 1.8k–3.1k against 0.8k–1.7k)',
     lambda: '(peak ' + ' against '.join(
         span([max(b['max'] for b in mech()[c]['knob']) / 1000 for c in after_switch(p)], '{:.1f}k'.format)
         for p in ('bc', 'cut')) + ')'),
    ('Appendix B: 38M', None, lambda: autograd_row("38M (trained: `hybrid_cutk_f0.3`'s final weights)", 'test_bc38', 128, 2048)),
    ('Appendix B: 7.3M', None, lambda: autograd_row('7.3M (init)', 'test_bc7', 64, 1024)),
    ('Appendix B: 243M', None, lambda: autograd_row('243M (init)', 'prof_bc243', 256)),
    ('Appendix B: plain GPT', None, lambda: autograd_row('plain GPT, 29M (init)', 'test_plain3', 128, 2048)),
]


# ---------------------------------------------------------------------------------------------------- record checks

def record_checks():
    rows = []
    for name, d in sorted(rescores().items()):
        if 'error' in d:
            rows.append((name, 'FAIL', f'error: {d["error"]}'))
            continue
        problems = []
        for split in ('val', 'test'):
            logged, rescored = d['logged'].get(split), d['rescored'].get(split)
            if logged is not None and rescored is not None and abs(logged - rescored) > 1e-4:
                problems.append(f'{split}: logged {logged:.6f}, re-scored {rescored:.6f}')
        if 'official_val' in d and abs(d['official_val'] - d['logged']['val']) > 1e-4:
            problems.append(f'official scorer {d["official_val"]:.6f} vs logged {d["logged"]["val"]:.6f}')
        if d['data_dir'] == '/data/data':
            if d['split_token_sha256'].get('val') != VAL_SHA:
                problems.append('validation split is not the pinned one')
            if d['split_token_sha256'].get('test') not in (None, TEST_SHA):
                problems.append('test split differs')
        detail = '; '.join(problems) or (f'val {d["rescored"].get("val", float("nan")):.5f}, test '
                                        f'{d["rescored"].get("test", float("nan")):.5f}'
                                        + (f', official {d["official_val"]:.5f}' if 'official_val' in d else ''))
        rows.append((name, 'FAIL' if problems else 'PASS', detail))
    return rows


def data_dir(run):
    """The run's data folder: the last --data-dir of its logged command."""
    log = HERE / 'logs' / f'{run}.log'
    if not log.exists():
        return None
    args = log.open(errors='ignore').readline().split()
    dirs = [args[i + 1] for i, a in enumerate(args[:-1]) if a == '--data-dir']
    return dirs[-1] if dirs else None


def reruns():
    """Finished runs whose config.json is identical (same code path, settings and seed): how far apart they land."""
    groups = {}
    for path in sorted((ROOT / 'results').glob('*/config.json')):
        config = json.loads(path.read_text())
        config.pop('output', None)
        config['data_dir'] = data_dir(path.parent.name)  # config.json does not record it; the command does
        config.setdefault('arch', 'noise')  # recorded by the scripts only once a second architecture existed
        groups.setdefault(json.dumps(config, sort_keys=True), []).append(path.parent.name)
    rows = []
    for runs in groups.values():
        finished = [r for r in runs if (ROOT / 'results' / r / 'result.json').exists()]
        if len(finished) > 1:
            tests = [result(r).get('test_loss') for r in finished]
            vals = [result(r)['val_loss'] for r in finished]
            rows.append((' = '.join(finished), ', '.join(f'{v:.4f}' for v in vals),
                         ', '.join(f'{t:.4f}' for t in tests if t is not None)))
    return rows


def main():
    lines = ['# Verification\n', 'Generated by `ledger/verify.py` from the run records; do not edit by hand.\n',
             '## Claims\n', '| where | printed | recomputed | status |', '|---|---|---|---|']
    counts = {'PASS': 0, 'FAIL': 0, 'PENDING': 0}
    flat = lambda s: s.replace('\n', '⏎').replace('|', '\\|')
    for where, printed, compute, document, doc_name in ([(*c, DRAFT, 'the draft') for c in CLAIMS]
                                                     + [(*c, README, 'the README') for c in README_CLAIMS]):
        try:
            value = compute()
        except Pending as missing:
            counts['PENDING'] += 1
            lines.append(f'| {where} | {flat(printed or "")} | waiting on {missing} | PENDING |')
            continue
        if printed is None:  # generated rows: the paper must print exactly this
            printed = value
        ok = value.strip() == printed.strip() and printed in document
        status = 'PASS' if ok else 'FAIL'
        if value.strip() == printed.strip() and printed not in document:
            status = 'FAIL'
            value = value + f' (not found in {doc_name})'
        counts[status] += 1
        lines.append(f'| {where} | {flat(printed)} | {flat(value)} | {status} |')
    lines += ['', '## Records: every checkpoint re-scored from its weights alone\n', '| run | status | detail |',
              '|---|---|---|']
    record_rows = record_checks()
    for entry, (_, run) in ENTRY_LOGS.items():  # each official log: its run's own result, from a re-scored checkpoint
        try:
            logged, ran = entry_log(entry), result(run)
            problems = [f'the run has {ran.get(k)} for {k}' for k in ('val_loss', 'train_seconds')
                        if logged.get(k) != ran.get(k)]
            checked = rescores().get(run)
            if checked is None or 'final.pt' not in checked.get('sha256', {}):
                problems.append('its checkpoint is not re-scored')
            else:
                for label, value in (('re-scored', checked['rescored'].get('val')),
                                     ('official scorer', checked.get('official_val'))):
                    if value is None or abs(value - logged['val_loss']) > 1e-6:
                        problems.append(f'{label} val {value}')
            detail = f'val {logged["val_loss"]:.5f} in {logged["train_seconds"]} s'
            if checked and 'final.pt' in checked.get('sha256', {}):
                detail += (f'; checkpoint sha256 {checked["sha256"]["final.pt"][:16]}…, re-scored '
                           f'{checked["rescored"]["val"]:.5f}, official scorer {checked.get("official_val", float("nan")):.5f}')
            record_rows.append((f'entry {entry} = run {run}', 'FAIL' if problems else 'PASS',
                                detail + (': ' + '; '.join(problems) if problems else '')))
        except Pending as missing:
            record_rows.append((f'entry {entry} = run {run}', 'PENDING', f'waiting on {missing}'))
    for name, status, detail in record_rows:
        lines.append(f'| {name} | {status} | {detail} |')
    lines += ['', '## Reruns: identical settings and seed, run at different times\n', '| runs | val | test |', '|---|---|---|']
    lines += [f'| {runs} | {vals} | {tests} |' for runs, vals, tests in reruns()]
    failed_records = sum(status == 'FAIL' for _, status, _ in record_rows)
    entries = [r for r in record_rows if r[0].startswith('entry ')]
    rescored = [r for r in record_rows if not r[0].startswith('entry ')]
    agree = lambda rows: sum(r[1] == 'PASS' for r in rows)
    lines.insert(2, f'**{counts["PASS"]} claims pass, {counts["FAIL"]} fail, {counts["PENDING"]} pending; '
                    f'{agree(rescored)} of {len(rescored)} re-scored checkpoints agree with their logs; '
                    f'{agree(entries)} of {len(entries)} entry logs match their runs and re-scored checkpoints.**\n')
    (HERE / 'verification.md').write_text('\n'.join(lines) + '\n')
    print(lines[2])
    sys.exit(1 if counts['FAIL'] or failed_records else 0)


if __name__ == '__main__':
    main()
