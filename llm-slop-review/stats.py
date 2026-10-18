"""Significance of the paper's seed comparisons: Welch's t-test (unequal variances, two-sided) on held-out test loss,
from the runs collect.py fetched into results/<run>/. Cells that have not finished are skipped.

    uv run --with scipy python llm-slop-review/stats.py
"""
import json
from pathlib import Path
import statistics

from scipy import stats

RES = Path(__file__).resolve().parent.parent / 'results'
CELLS = {
    'Boost → Cut': [f'bc38f_s{s}' for s in range(3)],
    'Boost → Cut, Boost keys sampled': [f'bc38_s{s}' for s in range(3)],
    'Cut alone': ['bc38_sw0', 'bc38_sw0_s1', 'bc38_sw0_s2'],
    'Boost alone': [f'bc38f_sw100_s{s}' for s in range(3)],
    'Dust record': [f'dust38_s{s}' for s in range(3)],
    'backprop': [f'bp38_s{s}' for s in range(3)],
}
PAIRS = [('Boost → Cut', 'Dust record'), ('Boost → Cut', 'Cut alone'), ('Boost → Cut', 'Boost alone'),
         ('Boost → Cut', 'Boost → Cut, Boost keys sampled'), ('Boost → Cut', 'backprop')]


def losses(runs, key='test_loss'):
    out = []
    for run in runs:
        path = RES / run / 'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            if key in result:
                out.append(result[key])
    return out


def main():
    print('| cell | n | test loss, mean ± sd |\n|---|---|---|')
    for name, runs in CELLS.items():
        xs = losses(runs)
        if xs:
            sd = f' ± {statistics.stdev(xs):.3f}' if len(xs) > 1 else ''
            print(f'| {name} | {len(xs)} | {statistics.mean(xs):.4f}{sd} |')
    print('\n| comparison | difference | 95% CI (Welch) | t | dof | p |\n|---|---|---|---|---|---|')
    for a, b in PAIRS:
        xa, xb = losses(CELLS[a]), losses(CELLS[b])
        if len(xa) < 2 or len(xb) < 2:
            continue
        test = stats.ttest_ind(xa, xb, equal_var=False)
        ci = test.confidence_interval(0.95)
        diff = statistics.mean(xa) - statistics.mean(xb)
        print(f'| {a} − {b} | {diff:+.4f} | [{ci.low:+.4f}, {ci.high:+.4f}] | {test.statistic:.2f} | {test.df:.1f} '
              f'| {test.pvalue:.2g} |')


if __name__ == '__main__':
    main()
