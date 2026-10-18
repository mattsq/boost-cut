# Ledgers: how to check every number in the paper

The ledgers tie each number in `llm-slop-review/draft.md` to the runs that produced it, each run to its exact command, code,
data, weights and cost, and each cost to Modal's own billing records. They are built from the project's records by
scripts that anyone can rerun.

## What is here

| file | what it records | built by |
|---|---|---|
| `verification.md` | every number the paper prints, recomputed from the run records (pass, fail or pending); every checkpoint's re-score against its log | `verify.py` |
| `runs.json` | every run behind the paper: its role (table and row), command, Modal app ID, start and end, config, result, billed cost, re-score and code version | `build.py` |
| `rescore.jsonl` | every finished run's checkpoint: the sha256 of `final.pt`, `config.json`, `result.json` and `metrics.jsonl`; validation and test loss re-scored from the weights alone; the official scorer's validation loss for the benchmark model; the token hash of each split it scored | `noise/analysis/ledger_rescore.py`, on Modal |
| `jobs.csv` | every job of the project, development included: start, end, GPUs, Modal app ID and billed cost | `build.py` |
| `compute.md` | total compute, reconciled with Modal's billing | `build.py` |
| `billing_export.json` | Modal's hourly billing export for the workspace (`modal billing report --json`), unedited | `build.py` |
| `logs/<run>.log` | each run's full log from the Modal volume; its first line is the exact command, which also names the data folder (`config.json` does not) | `build.py` |
| `provenance/<run>/job.json` | for jobs from 2026-10-17 18:12 PDT on: a sha256 manifest of the exact code in the container (the snapshot itself, `code.tar.gz`, stays on the Modal volume), the data files' sha256, the GPUs and driver, and the torch, CUDA and Python versions | `modal_app.py` (`tools/` in the release), on Modal |

## Checking the paper's numbers

```bash
python3 ledger/verify.py          # recomputes every claim from results/, logs/ and rescore.jsonl
```

Each claim names where it is printed, the printed text, and how it is recomputed: means and standard deviations over
seeds, differences between runs, ratios of counted FLOPs, ranges over checkpoints. A claim passes only if the
recomputed text matches the printed text and that text appears in the draft. Rounding is part of the check. The first
run of the verifier caught 11 discrepancies in the draft, all now fixed:
- three roundings (2.3× for 2.4×, 0.024 for 0.025, +0.194 for +0.195);
- one stale row (Cut alone before its seeds finished);
- mechanism-table rows paired by step rather than by matched loss, and two of their unit counts;
- a saliency range;
- an appendix column that mixed two route budgets.

## Checking the weights

Every finished run's `final.pt` is hashed and re-scored on Modal from the checkpoint alone. A fresh model is built
from the run's `config.json`, and no training code is involved. Validation and test loss are recomputed with
`utils/eval.py`'s exact computation. For runs of the benchmark's own model, `utils/eval.py` itself (the official
scorer) also scores validation. `verification.md` lists each run.

So far every re-scored checkpoint reproduces its logged losses to within 1e-7. Single-GPU and 8-GPU evaluation sum
in different orders, which accounts for that residual.

With the checkpoints in hand, anyone can repeat this:

```bash
shasum -a 256 final.pt                                     # against rescore.jsonl
uv run utils/eval.py final.pt                              # the official validation score (benchmark model)
python analysis/ledger_rescore.py --runs <run> --runs-dir <dir> --data-dir data --out mine.jsonl
```

## Checking the data

- **Validation** is the benchmark's pinned split: token sha256 `5fdb9242…`, checked by `utils/prepare_data.py`.
- **Test** is the stream one pool past validation, as the Dust repository defines it (`analysis/prepare_test.py`):
  token sha256 `cb7a29f0…`.
- Every re-score records the token hash of each split it scored, and the verifier checks them.
- From 18:12 PDT on, each job's provenance also records the sha256 of every data file on the volume.

## Checking the compute

`compute.md` sums Modal's billing export by app. `jobs.csv` attributes each app to a named job through the app ID
the launcher printed when it started the job (kept in `jobs.csv`; the launch logs are in `inputs/`). Apps that match no named job are listed with
their cost: manual runs, early tests and a peer session sharing the workspace. Billing is the authority. The
queue's clock (start to end × GPUs) is a cross-check.

## How much of each run's code is pinned

`runs.json` gives each run a code level:

| level | meaning | runs |
|---|---|---|
| `snapshot` | the container's exact code, archived and hashed by the launcher | every job launched from 2026-10-17 18:12 PDT |
| `reconstructed` | the training script's exact content is known from file timestamps and a hashed copy: `train/boostcut/run.py` was unchanged from 16:29:52 to 17:41:48 PDT, and its copy `submission/val_run_boostcut.py` (sha256 `6e02c1f0…`) was hashed at 17:16 | the three headline seeds `bc38f_s0–s2`, `bc7_s0`, `bcts_s0`, `bc38_dfa` |
| `command` | only the logged command and `config.json`, which records every setting the run used | earlier runs |

Between the pinned versions and today's code, the scripts only gained options whose defaults keep the old behavior:
- `--lr-mult` (default 1.0) in `run.py` and `backprop_preset.py`;
- `--arch` (default `noise`) in `backprop_preset.py`.

The oracle scripts (`analysis/oracle_hybrid.py`, `analysis/oracle_boost.py`) now take the record's model and optimizer setup from `train/boostcut/run.py`. That setup is the same code their logged runs used, apart from formatting and comments.

`config.json` records each run's effective settings, including those that later became defaults (for example
`boost_exact_keys`). A rerun with today's code and a run's recorded settings reproduces it within seed noise. For
example, `bc38_bexact` and `bc38f_s0` share every setting and seed and were run at different times: test 3.937
against 3.937, and validation 3.947 against 3.945.

## What the ledgers cannot show

- **Wall-clock** depends on the node. Identical settings took 394 s and 456 s on two runs. Counted FLOPs
  (`analysis/flops.py`) are the stable measure.
- **GPU kernels are not bitwise deterministic**, so a rerun matches within seed noise, not to the last digit.
- **The Modal app pages** linked in `runs.json` are visible to the workspace only. The billing export and logs
  carry the same information in the open.

## In this repository

- `billing_export.json` keeps every Modal billing row but not the app-name field; rows match jobs by app ID.
- `jobs.csv` and `inputs/queue.log` cover the paper's runs (from 2026-10-17 12:20 PDT). The totals in `compute.md` also count the earlier development jobs, which the billing export holds by app ID.
- `provenance/*/job.json` lists hashes for the files released here; `code_sha256` still covers the whole code tree that ran.
