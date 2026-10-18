"""Builds the verification ledgers from the project's own records (run from the project root).

Sources, all kept next to the ledgers so anyone can re-derive them:
- queue.log, side_queue.log, par.log: when each job started and ended (local time, PDT);
- localruns/<job>.out: each job's Modal app ID (the launcher prints its URL);
- Modal's billing export (`modal billing report --json`, saved as billing_export.json): what each app cost;
- results/<run>/: each run's config.json, metrics.jsonl and result.json (fetched by collect.py);
- /data/logs/<run>.log on the Modal volume (saved under logs/): the exact command and the full output;
- /data/provenance/<run>/job.json (saved under provenance/): code and data hashes, GPUs, versions (jobs from
  2026-10-17 18:15 PDT on);
- /data/ledger/rescore.jsonl (saved as rescore.jsonl): checkpoint hashes and independent re-scores.

Writes jobs.csv (every job of the project, with its billed cost), runs.json (every run behind the paper, with
everything above joined) and compute.md (totals).

    python3 ledger/build.py [--refresh]
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'tools'))  # the release keeps collect.py under tools/
import collect  # noqa: E402  (the paper's run list: collect.RUNS)

PDT = timezone(timedelta(hours=-7))
# each launcher log and the date of its first line (the logs carry times only)
QUEUE_LOGS = {'queue.log': datetime(2026, 10, 16, tzinfo=PDT), 'side_queue.log': datetime(2026, 10, 16, tzinfo=PDT),
              'par.log': datetime(2026, 10, 17, tzinfo=PDT)}
H100_RATE = 3.95  # USD per H100-hour on Modal (0.001097 per second)
PAPER_START = datetime(2026, 10, 17, 12, 20, tzinfo=PDT)
ONE_GPU = ('test_', 'flops', 'mech', 'rescore', 'memory', 'oracle', 'oh_', 'clean_probe', 'ledger_', 'prof_')
NO_GPU = ('prepare_test', 'ts_prep', 'prep')
EXTRA_RUNS = ['oracle_hybrid', 'oh_base', 'oh_route128', 'oh_route64', 'mech38', 'flops38b', 'flops243', 'memory38',
              'test_bc38', 'test_bc7', 'test_plain3', 'prof_bc243', 'fid_plain', 'fid_noise', 'rescore_bc38f', 'prepare_test', 'ts_prep', 'bc243_s0']


# Code versions of runs launched before per-job snapshots existed, reconstructed from file timestamps: the training
# script's content is known for a window when its file was unchanged and a copy was hashed inside that window.
CODE_WINDOWS = [
    # (script, first moment, last moment, sha256, how the content is known)
    ('train/boostcut/run.py', datetime(2026, 10, 17, 16, 29, 52, tzinfo=PDT), datetime(2026, 10, 17, 17, 41, 48, tzinfo=PDT),
     '6e02c1f075ca04f24196cba14ca0c1c2b0312a556381c8ccec6e71317ba9ac23',
     'byte-identical to submission/val_run_boostcut.py, copied and hashed at 17:16 with the file unchanged since 16:29:52; '
     'the next edit (17:41:48) only added --lr-mult, default 1.0'),
    ('train/sub_boost/run.py', datetime(2026, 10, 17, 9, 12, 14, tzinfo=PDT), datetime(2100, 1, 1, tzinfo=PDT),
     'd76005f87bfd92e4a5a3a3e2cc40c0c6daedbf01e71e2faf760ecaafe8554b9d',
     'the Track 2 entry: unchanged since 09:12:14, before its run, and byte-identical to entries/track2_boost/run.py'),
]
SNAPSHOTS_FROM = datetime(2026, 10, 17, 18, 11, 52, tzinfo=PDT)  # modal_app.py writes provenance/<job>/ from here on


def code_version(name, start, command):
    """How the exact code of a run is known: its own snapshot, a reconstructed version, or only its command."""
    if start is None:
        return dict(level='unknown')
    if start >= SNAPSHOTS_FROM:
        return dict(level='snapshot', where=f'provenance/{name}/code.tar.gz (sha256 manifest in job.json)')
    for script, first, last, sha, how in CODE_WINDOWS:
        if command and f' {script} ' in f' {command} ' and first <= start < last:
            return dict(level='reconstructed', script=script, sha256=sha, how=how)
    return dict(level='command', note='the logged command and config.json; the script has since gained only '
                                     'options whose defaults keep its behavior (see ledger/README.md)')


def modal(*args):
    env = dict(os.environ)
    return subprocess.run(['modal', *args], env=env, capture_output=True, text=True)


def fetch(remote, local, refresh):
    if local.exists() and not refresh:
        return local.exists()
    local.parent.mkdir(parents=True, exist_ok=True)
    modal('volume', 'get', 'noise-forward', remote, str(local), '--force')
    return local.exists()


def queue_jobs():
    """(name, start, end, log) from the launcher logs, with day rollovers counted from each log's first day."""
    jobs = []
    for log_name, first_day in QUEUE_LOGS.items():
        path = ROOT / log_name
        if not path.exists():
            continue
        day, last, starts = 0, None, {}
        for line in path.read_text().splitlines():
            m = re.match(r'(\d\d):(\d\d):(\d\d) (start|end) (\S+)', line)
            if not m:
                continue
            seconds = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            if last is not None and seconds < last - 60:
                day += 1
            last = seconds
            when = first_day + timedelta(days=day, seconds=seconds)
            if m[4] == 'start':
                starts[m[5]] = when
            elif m[5] in starts:
                jobs.append((m[5], starts.pop(m[5]), when, path.name))
        jobs += [(name, start, None, path.name) for name, start in starts.items()]  # still running
    return jobs


def app_ids():
    ids = {}
    for path in (ROOT / 'localruns').glob('*.out'):
        m = re.search(r'ap-[A-Za-z0-9]{20,}', path.read_text(errors='ignore'))
        if m:
            ids[path.stem] = m[0]
    return ids


def gpus(name):
    if name.startswith(NO_GPU):
        return 0
    return 1 if name.startswith(ONE_GPU) else 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Re-fetch logs, provenance, re-scores and billing.')
    parser.add_argument('--billing-start', default='2026-10-15')
    args = parser.parse_args()

    billing_path = HERE / 'billing_export.json'
    if args.refresh or not billing_path.exists():
        out = modal('billing', 'report', '--start', args.billing_start, '--show-resources', '--resolution', 'h',
                    '--json')
        rows = json.loads(out.stdout)
        billing_path.write_text(json.dumps(rows, indent=1) + '\n')
    billing = json.loads(billing_path.read_text())
    cost = defaultdict(lambda: defaultdict(float))
    for row in billing:
        cost[row['object_id']][row['resource']] += float(row['cost'])

    ids = app_ids()
    jobs = queue_jobs()
    attributed = set()
    with (HERE / 'jobs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['job', 'queue', 'phase', 'start_pdt', 'end_pdt', 'gpus', 'wall_hours', 'gpu_hours_by_clock',
                         'modal_app_id', 'billed_h100_usd', 'billed_h100_hours', 'billed_other_usd'])
        for name, start, end, queue in sorted(jobs, key=lambda j: j[1]):
            app = ids.get(name, '')
            attributed.add(app)
            wall = ((end - start).total_seconds() / 3600) if end else None
            h100 = cost[app].get('H100', 0.0) if app else 0.0
            other = sum(v for k, v in cost[app].items() if k != 'H100') if app else 0.0
            writer.writerow([name, queue, 'paper' if start >= PAPER_START else 'development',
                             start.isoformat(timespec='seconds'), end.isoformat(timespec='seconds') if end else '',
                             gpus(name), f'{wall:.4f}' if wall is not None else '',
                             f'{wall * gpus(name):.3f}' if wall is not None else '', app, f'{h100:.4f}',
                             f'{h100 / H100_RATE:.3f}', f'{other:.4f}'])

    # every billed app: attributed to a named job, or listed as unattributed (manual runs and the peer session)
    totals = defaultdict(float)
    for app, resources in cost.items():
        for resource, usd in resources.items():
            totals[resource, app in attributed] += usd
    paper_h100 = sum(cost[ids[n]].get('H100', 0.0) for n, s, e, q in jobs if n in ids and s >= PAPER_START)
    by_clock = sum(((e - s).total_seconds() / 3600) * gpus(n) for n, s, e, q in jobs if e and s >= PAPER_START)
    unattributed = sorted({row['object_id'] for row in billing} - attributed)
    with (HERE / 'compute.md').open('w') as f:
        h100 = sum(v for (r, _), v in totals.items() if r == 'H100')
        f.write('# Compute ledger\n\n')
        f.write(f'Modal billing export: {len(billing)} hourly rows, {len({r["object_id"] for r in billing})} apps, '
                f'{min(r["interval_start"] for r in billing)} to {max(r["interval_start"] for r in billing)} UTC '
                '(`billing_export.json`).\n\n')
        f.write('| | USD | H100-hours (at $3.95) |\n|---|---|---|\n')
        f.write(f'| H100, all apps | {h100:,.2f} | {h100 / H100_RATE:,.1f} |\n')
        f.write(f'| H100, apps matched to a named job | {totals["H100", True]:,.2f} | '
                f'{totals["H100", True] / H100_RATE:,.1f} |\n')
        f.write(f'| H100, unmatched apps ({len(unattributed)}: manual runs, tests, the peer session) | '
                f'{totals["H100", False]:,.2f} | {totals["H100", False] / H100_RATE:,.1f} |\n')
        f.write(f'| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | {paper_h100:,.2f} | '
                f'{paper_h100 / H100_RATE:,.1f} |\n')
        other = sum(v for (r, _), v in totals.items() if r != 'H100')
        f.write(f'| CPU, memory and egress, all apps | {other:,.2f} | |\n')
        f.write(f'| **total billed** | **{h100 + other:,.2f}** | |\n\n')
        f.write(f'Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock '
                f'gives {by_clock:,.1f} H100-hours against {paper_h100 / H100_RATE:,.1f} billed. The clock also counts '
                'container scheduling and start-up; billing is the authority.\n')
        f.write('\nBilling lags by about an hour, so jobs that ended within the last hour are not fully billed yet.\n')

    # every run behind the paper
    rescore = {}
    fetch('ledger/rescore.jsonl', HERE / 'rescore.jsonl', True)
    if (HERE / 'rescore.jsonl').exists():
        for line in (HERE / 'rescore.jsonl').read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rescore[record['name']] = record
    times = {n: (s, e) for n, s, e, q in jobs}
    runs = {}
    for name in list(collect.RUNS) + EXTRA_RUNS:
        table, label = collect.RUNS.get(name, ('analysis', name))
        record = dict(name=name, table=table, label=label, modal_app_id=ids.get(name),
                      modal_app_url=f'https://modal.com/apps/<workspace>/main/{ids[name]}' if name in ids else None)
        if name in times:
            s, e = times[name]
            record.update(start_pdt=s.isoformat(timespec='seconds'), end_pdt=e.isoformat(timespec='seconds') if e else None)
        if name in ids:
            record.update(billed_h100_usd=round(cost[ids[name]].get('H100', 0.0), 4))
        result_dir = ROOT / 'results' / name
        for kind in ('config', 'result'):
            path = result_dir / f'{kind}.json'
            if path.exists():
                record[kind] = json.loads(path.read_text())
        if fetch(f'logs/{name}.log', HERE / 'logs' / f'{name}.log', args.refresh):
            first = (HERE / 'logs' / f'{name}.log').open(errors='ignore').readline().strip()
            record['command'] = first[2:] if first.startswith('$ ') else None
        if fetch(f'provenance/{name}/job.json', HERE / 'provenance' / name / 'job.json', args.refresh):
            job = json.loads((HERE / 'provenance' / name / 'job.json').read_text())
            record['provenance'] = {k: job.get(k) for k in ('code_sha256', 'data_files', 'gpus', 'torch', 'cuda',
                                                             'python', 'start_utc', 'end_utc', 'exit_code')}
        if name in rescore:
            record['rescore'] = rescore[name]
        start = times.get(name, (None, None))[0]
        record['code'] = code_version(name, start, record.get('command'))
        runs[name] = record
    (HERE / 'runs.json').write_text(json.dumps(runs, indent=1, default=str) + '\n')
    print(f'jobs.csv: {len(jobs)} jobs; runs.json: {len(runs)} runs '
          f'({sum("result" in r for r in runs.values())} finished, {len(rescore)} re-scored); compute.md')


if __name__ == '__main__':
    main()
