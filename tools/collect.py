"""Collects the paper runs' results from the Modal volume (results/<run>/) and prints the tables, mean +- sd over seeds.

    python3 collect.py
"""
import json
import os
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'

# run name -> (table, row label)
RUNS = {
    **{f'bc38f_s{s}': ('main', 'Boost → Cut (all keys exact)') for s in (0, 1, 2)},
    **{f'bc38_s{s}': ('main', 'Boost → Cut, Boost keys sampled (the ablations\' base)') for s in (0, 1, 2)},
    **{f'bp38_s{s}': ('main', 'backprop (same optimizer)') for s in (0, 1, 2)},
    **{f'dust38_s{s}': ('main', 'Dust record (16k)') for s in (0, 1, 2)},
    'bc38_fg128': ('main', 'local losses + forward gradient (128 directions)'),
    'bp38_curve': ('curve', 'backprop, seed 0 with validation curve'),
    'bc38_dfa': ('main', 'direct feedback alignment'),
    **{f'bc38_dfa_lr{m}': ('main', f'direct feedback alignment, learning rates x{m}') for m in ('0.3', '0.1', '0.03')},
    'bc38_fg128_lr0.5': ('main', 'local losses + forward gradient, learning rates x0.5'),
    **{f'bc38f_lr{m}': ('ablation', f'Boost → Cut, learning rates x{m} (seed 0)') for m in ('0.5', '2')},
    'bc38_sw100': ('ablation', 'Boost alone (switch 100%, Boost keys sampled)'),
    **{r: ('ablation', 'Cut alone (switch 0%)') for r in ('bc38_sw0', 'bc38_sw0_s1', 'bc38_sw0_s2')},
    **{f'bc38f_sw100_s{s}': ('ablation', 'Boost alone (switch 100%, exact keys)') for s in (0, 1, 2)},
    **{f'bp38_lr{m}': ('main', f'backprop, learning rates x{m}') for m in ('0.5', '2', '4')},
    'bc38_sw50': ('ablation', 'switch 50%'),
    'bc38_k64': ('ablation', 'routes: top-64'),
    'bc38_k256': ('ablation', 'routes: top-256'),
    'bc38_kall': ('ablation', 'routes: all 2,048'),
    'bc38_ksampled': ('ablation', 'Cut keys sampled'),
    'bc38_bexact': ('ablation', 'Boost keys exact too (seed 0)'),
    'bc7_s0': ('scale', '7.3M Boost → Cut'), 'bp7_s0': ('scale', '7.3M backprop'), 'dust7_s0': ('scale', '7.3M Dust'),
    'bc243_s0': ('scale', '243M Boost → Cut (Boost keys sampled)'), 'bc243f_s0': ('scale', '243M Boost → Cut'), 'bp243_s0': ('scale', '243M backprop'),
    'dust38_p4k': ('scale', '38M Dust, 4k population'), 'dust243_p4k': ('scale', '243M Dust, 4k population'),
    'plain_bc38_s0': ('plain', 'Boost → Cut'), 'plain_bp38_s0': ('plain', 'backprop'), 'plain_dust38_s0': ('plain', 'Dust record'),
    'plain_bc38_kall': ('plain', 'Boost → Cut, all 2,048 units'),
    'boost_official_s0': ('speedrun', 'Boost speedrun entry (official log)'),
    'dust_speedrun_official': ('speedrun', 'Dust speedrun record, its official script, our hardware'),
    'bcts_s0': ('tinystories', 'Boost → Cut'), 'bpts_s0': ('tinystories', 'backprop'), 'dustts_s0': ('tinystories', 'Dust record'),
}


def fetch(run):
    target = OUT / run
    if (target / 'result.json').exists():
        return json.loads((target / 'result.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('result.json', 'config.json', 'metrics.jsonl'):
        subprocess.run(['modal', 'volume', 'get', 'noise-forward', f'runs/{run}/{name}', str(target / name), '--force'],
                       env=env, capture_output=True)
    path = target / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def fmt(values):
    if not values:
        return '–'
    if len(values) == 1:
        return f'{values[0]:.3f}'
    return f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f} (n={len(values)})'


def main():
    rows = {}
    for run, (table, label) in RUNS.items():
        result = fetch(run)
        if result is None:
            continue
        cell = rows.setdefault((table, label), dict(val=[], test=[], seconds=[]))
        cell['val'].append(result['val_loss'])
        if 'test_loss' in result:
            cell['test'].append(result['test_loss'])
        cell['seconds'].append(result['train_seconds'])
    for table in ('main', 'ablation', 'scale', 'tinystories', 'plain', 'speedrun'):
        print(f'\n### {table}\n\n| run | test loss | val loss | train time (s, 8xH100) |\n|---|---|---|---|')
        for (t, label), cell in rows.items():
            if t == table:
                secs = statistics.mean(cell['seconds'])
                print(f"| {label} | {fmt(cell['test'])} | {fmt(cell['val'])} | {secs:,.0f} |")


if __name__ == '__main__':
    main()
