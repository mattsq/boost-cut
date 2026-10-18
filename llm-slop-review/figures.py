"""Figures for the paper, from the runs collect.py fetched into results/<run>/ and the mechanism probe's log
(localruns/mech38.out, or ledger/logs/mech38.log). Runs that have not finished are left out of a figure, never invented.

    python3 collect.py && python3 llm-slop-review/figures.py   # writes paper/fig_*.png
"""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / 'results'
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
# categorical slots in fixed order (validated: scripts/validate_palette.js, light surface); every series is also
# labelled directly, since slots 3-5 sit below 3:1 against the surface
BOOSTCUT, DUST, BACKPROP, CUT, BOOST, FG, DFA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
SEQ, EPOCH = 2048, 4880

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'axes.titlesize': 10, 'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


def result(run):
    path = RES / run / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def curve(run):
    """[(training tokens, validation loss)], the final evaluation included; None if the run has not finished."""
    res = result(run)
    if res is None:
        return None
    points = {}
    for line in (RES / run / 'metrics.jsonl').read_text().splitlines():
        d = json.loads(line)
        if 'val_loss' in d:
            seqs = d['seqs'] if 'seqs' in d else d['step'] * 8
            points[seqs] = d['val_loss']
    points[EPOCH] = res['val_loss']
    return sorted((s * SEQ, v) for s, v in points.items())


def mean_curve(runs):
    curves = [c for c in map(curve, runs) if c]
    if not curves:
        return None, 0
    common = set.intersection(*(set(x for x, _ in c) for c in curves))
    return sorted((x, statistics.mean(dict(c)[x] for c in curves)) for x in common), len(curves)


def end_label(ax, points, text, color, dy=0.0):
    x, y = points[-1]
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center', color=INK2, fontsize=8.5)
    ax.plot([x], [y], 'o', color=color, ms=4, mec=SURFACE, mew=1.5, zorder=4)


def spread(ys, gap):
    """Label heights at least gap apart, as close to ys as possible (ys sorted ascending in the output order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = {}
    for i in order:
        placed[i] = max(ys[i], placed[order[order.index(i) - 1]] + gap) if order.index(i) else ys[i]
    shift = (sum(placed.values()) - sum(ys)) / len(ys)  # centre the block on the points
    return [placed[i] - shift for i in range(len(ys))]


def fig_curves():
    """Validation loss along the epoch, 38M: Boost -> Cut against its two phases alone, the Dust record and backprop."""
    series = [
        ('Boost → Cut', [f'bc38f_s{s}' for s in range(3)], BOOSTCUT, 0),
        ('Cut alone', ['bc38_sw0', 'bc38_sw0_s1', 'bc38_sw0_s2'], CUT, 6),
        ('Boost alone', [f'bc38f_sw100_s{s}' for s in range(3)], BOOST, 0),
        ('Dust record', [f'dust38_s{s}' for s in range(3)], DUST, -6),
        ('backprop', ['bp38_curve'], BACKPROP, 0),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ends = []
    for label, runs, color, _ in series:
        points, n = mean_curve(runs)
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot([x / 1e6 for x in xs], ys, color=color, label=label)
        ax.plot([xs[-1] / 1e6], [ys[-1]], 'o', color=color, ms=4, mec=SURFACE, mew=1.5, zorder=4)
        ends.append((xs[-1] / 1e6, ys[-1], f'{label} {ys[-1]:.3f}' + (f' (n={n})' if n > 1 else '')))
    for (x, y, text), y_text in zip(ends, spread([e[1] for e in ends], 0.08)):
        ax.annotate(text, (x, y), xytext=(x + 0.35, y_text), textcoords='data', va='center', color=INK2, fontsize=8.5,
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8, shrinkA=0, shrinkB=3))
    ax.axvline(0.3 * EPOCH * SEQ / 1e6, color=AXIS, lw=1, ls='--', zorder=1)
    ax.text(0.3 * EPOCH * SEQ / 1e6, 5.45, '  switch (30%)', color=MUTED, fontsize=8, va='top')
    ax.set_xlabel('training tokens (millions)')
    ax.set_ylabel('validation loss')
    ax.set_ylim(3.6, 5.5)
    ax.set_xlim(0, 13.5)
    ax.set_title('Learning curves, 38M (one epoch of 10M tokens)')
    fig.tight_layout()
    fig.savefig(HERE / 'fig_curves.png', dpi=200)
    plt.close(fig)


def mean_result(runs):
    rs = [r for r in map(result, runs) if r and 'test_loss' in r]
    if not rs:
        return None
    return statistics.mean(r['train_seconds'] for r in rs), statistics.mean(r['test_loss'] for r in rs)


def fig_time():
    """Test loss against training wall-clock (8×H100), 38M, seed means where there are seeds. Left: every method.
    Right: Boost -> Cut's design space (the ablations, all with Boost's keys sampled, against their own base)."""
    methods = [
        ('Boost → Cut', [f'bc38f_s{s}' for s in range(3)], BOOSTCUT, (6, -4)),
        ('Cut alone', ['bc38_sw0', 'bc38_sw0_s1', 'bc38_sw0_s2'], CUT, (6, 4)),
        ('Boost alone', [f'bc38f_sw100_s{s}' for s in range(3)], BOOST, (-6, 0)),
        ('local losses + forward gradient', ['bc38_fg128'], FG, (-6, 0)),
        ('direct feedback alignment', ['bc38_dfa'], DFA, (-6, 0)),
        ('Dust record', [f'dust38_s{s}' for s in range(3)], DUST, (-6, 0)),
        ('backprop', [f'bp38_s{s}' for s in range(3)], BACKPROP, (6, 0)),
    ]
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for label, runs, color, (dx, dy) in methods:
        point = mean_result(runs)
        if point is None:
            continue
        ax.plot(*point, 'o', color=color, ms=6, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(label, point, xytext=(dx, dy), textcoords='offset points', va='center', color=INK2, fontsize=8,
                    ha='left' if dx > 0 else 'right')
    ax.set_xscale('log')
    ax.set_xlim(15, 10000)
    ax.set_xlabel('training time, seconds on 8×H100 (log)')
    ax.set_ylabel('test loss')
    ax.set_title('Every method')
    routes = [('top 64', ['bc38_k64'], (7, 0), 'left'),
              ('top 128 (30% switch)', [f'bc38_s{s}' for s in range(3)], (7, 3), 'left'),
              ('top 256', ['bc38_k256'], (0, -12), 'center'), ('all 2,048', ['bc38_kall'], (0, 9), 'center')]
    pts = [(label, mean_result(runs), offset, ha) for label, runs, offset, ha in routes]
    pts = [row for row in pts if row[1]]
    if pts:
        zoom.plot([p[0] for _, p, _, _ in pts], [p[1] for _, p, _, _ in pts], color=BOOSTCUT, marker='o', ms=5,
                  mec=SURFACE, mew=1.5, zorder=3)
        for label, p, offset, ha in pts:
            zoom.annotate(label, p, xytext=offset, textcoords='offset points', ha=ha, va='center', color=INK2,
                          fontsize=8)
    for label, runs, offset, ha in (('switch 0% (Cut alone)', ['bc38_sw0'], (7, 0), 'left'),
                                    ('switch 50%', ['bc38_sw50'], (0, 9), 'center')):
        point = mean_result(runs)
        if point:
            zoom.plot(*point, 'D', color=CUT, ms=5, mec=SURFACE, mew=1.2, zorder=3)
            zoom.annotate(label, point, xytext=offset, textcoords='offset points', ha=ha, va='center', color=INK2,
                          fontsize=8)
    zoom.set_xlabel('training time, seconds on 8×H100')
    zoom.set_title('Boost → Cut: routes and switch point')
    zoom.margins(x=0.12, y=0.18)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_time.png', dpi=200)
    plt.close(fig)


def params(n_layer, width, vocab=4096):
    """The benchmark GPT's parameter count: embeddings, head, blocks, value embeddings on half the blocks."""
    return 2 * vocab * width + n_layer * 12 * width ** 2 + (n_layer // 2) * vocab * width


def fig_scale():
    """Test loss against model size (the Dust paper's ladder, 10M tokens)."""
    sizes = {'7m': params(4, 256), '38m': params(8, 512), '243m': params(16, 1024)}
    series = [
        ('Boost → Cut', {'7m': ['bc7_s0'], '38m': [f'bc38f_s{s}' for s in range(3)], '243m': ['bc243f_s0']}, BOOSTCUT),
        ('Dust record', {'7m': ['dust7_s0'], '38m': [f'dust38_s{s}' for s in range(3)]}, DUST),
        ('Dust, 4k population', {'38m': ['dust38_p4k'], '243m': ['dust243_p4k']}, DUST),
        ('backprop', {'7m': ['bp7_s0'], '38m': [f'bp38_s{s}' for s in range(3)], '243m': ['bp243_s0']}, BACKPROP),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for label, cells, color in series:
        xs, ys = [], []
        for size, runs in cells.items():
            rs = [r for r in map(result, runs) if r and 'test_loss' in r]
            if rs:
                xs.append(sizes[size])
                ys.append(statistics.mean(r['test_loss'] for r in rs))
        if not xs:
            continue
        ls = '--' if '4k' in label else '-'
        ax.plot(xs, ys, ls, color=color, marker='o', ms=5, mec=SURFACE, mew=1.5)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points', va='center', color=INK2,
                    fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(list(sizes.values()), ['7.3M', '38M', '243M'])
    ax.minorticks_off()
    ax.set_xlim(5e6, 9e8)
    ax.set_xlabel('parameters (log scale)')
    ax.set_ylabel('test loss')
    ax.set_title('Scale, 10M training tokens')
    fig.tight_layout()
    fig.savefig(HERE / 'fig_scale.png', dpi=200)
    plt.close(fig)


def fig_routes():
    """Route fidelity at matched validation loss, with and without the head start (analysis/clean_probe.py)."""
    probe = {}
    log = next(p for p in (ROOT / 'localruns' / 'mech38.out', ROOT / 'ledger' / 'logs' / 'mech38.log') if p.exists())
    for line in log.read_text(errors='ignore').splitlines():
        line = line.strip()
        if line.startswith('{"bc_') or line.startswith('{"cut_'):
            probe.update(json.loads(line))
    vals = {}
    for run, prefix in (('bc38_s1', 'bc'), ('bc38_sw0', 'cut')):
        for line in (RES / run / 'metrics.jsonl').read_text().splitlines():
            d = json.loads(line)
            if 'val_loss' in d:
                vals[f"{prefix}_{d['seqs']}"] = d['val_loss']
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, k in zip(axes, ('64', '128', '256')):
        for prefix, label, color in (('bc', 'with head start', BOOSTCUT), ('cut', 'without', CUT)):
            pts = sorted((vals[name], min(rec['routes'][k]['token_cos'].values()))
                         for name, rec in probe.items() if name.startswith(prefix + '_') and name in vals)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, marker='o', ms=3.5, mec=SURFACE, mew=1, label=label)
        ax.invert_xaxis()
        ax.set_title(f'top {k} of 2,048 units')
        ax.set_xlabel('validation loss')
    axes[0].set_ylabel('min per-token cosine\n(route-restricted vs full)')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_routes.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    for make in (fig_curves, fig_time, fig_scale, fig_routes):
        try:
            make()
            print('wrote', make.__name__)
        except FileNotFoundError as missing:
            print('skipped', make.__name__, '(missing', Path(missing.filename).parent.name + ')')
