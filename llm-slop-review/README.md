# Reproducing the paper

Every number in `draft.md` comes from a named run below. Each run writes `config.json`, `metrics.jsonl`,
`result.json` (validation and test loss, training seconds) and `final.pt` to its output folder. `collect.py`
gathers the results into tables, `llm-slop-review/figures.py` draws the figures and `llm-slop-review/stats.py` runs the significance
tests.

Paths are relative to `noise/` (the benchmark repository, with this work's files added). Every run uses one 8×H100
node except where marked "1 GPU".

## Data

```bash
uv run utils/prepare_data.py                                   # the benchmark's train.pt and val.pt (pinned hashes)
python analysis/prepare_test.py --data-dir data                # test.pt: the stream one pool past val (sha256 cb7a29f0…)
python analysis/prepare_tinystories.py --data-dir data/tinystories
```

## Runs

`B` below stands for `torchrun --standalone --nproc_per_node=8`. Every training command also takes
`--data-dir data --output runs/<name>`.

| table / figure | runs | command |
|---|---|---|
| Table 1, Fig. 1–2: Boost → Cut | `bc38f_s{0,1,2}` | `B train/boostcut/run.py --preset 38m --seed S --eval-every 488 --test-file test.pt` |
| Table 1: backprop | `bp38_s{0,1,2}`; LR sweep `bp38_lr{0.5,2,4}`; curve `bp38_curve` | `B analysis/backprop_preset.py --preset 38m --seed S --test-file test.pt [--lr-mult M] [--eval-every 488]` |
| Table 1: Dust record | `dust38_s{0,1,2}` | `B analysis/dust_preset.py --preset 38m --seed S --eval-every 61 --test-file test.pt` |
| Table 1: forward gradient | `bc38_fg128`, `bc38_fg128_lr0.5` | `B train/boostcut/run.py --switch 1.0 --boost-probes 128 --boost-keys sampled [--lr-mult 0.5] --test-file test.pt` |
| Table 1: DFA | `bc38_dfa`, `bc38_dfa_lr{0.3,0.1,0.03}` | `B train/boostcut/run.py --switch 0 --estimator dfa [--lr-mult M] --test-file test.pt` |
| Table 1–2: Cut alone | `bc38_sw0`, `bc38_sw0_s{1,2}` | `B train/boostcut/run.py --switch 0 --seed S --test-file test.pt` |
| Table 1–2: Boost alone | `bc38f_sw100_s{0,1,2}` | `B train/boostcut/run.py --switch 1.0 --seed S --test-file test.pt` |
| Table 2: base (Boost keys sampled) | `bc38_s{0,1,2}` | `B train/boostcut/run.py --boost-keys sampled --seed S --test-file test.pt` |
| Table 2: ablations | `bc38_sw50`, `bc38_k64`, `bc38_k256`, `bc38_kall`, `bc38_ksampled`, `bc38_sw100` | as the base, plus `--switch 0.5`, `--route-k 64`, `--route-k 256`, `--route-k 2048`, `--keys sampled`, `--switch 1.0` |
| Table 2: learning rates | `bc38f_lr{0.5,2}` | `B train/boostcut/run.py --lr-mult M --test-file test.pt` |
| Section 4.4: oracles (1 GPU) | `oracle_hybrid`, `oh_base`, `oh_route128`, `oh_route64` | `python analysis/oracle_hybrid.py --switches 0.3,0.5,0.7 --phase2 attn,full --ends [--route-k K]` |
| Table 3, Fig. 3: scale | `bc7_s0`, `bc243f_s0` (and `bc243_s0`, Boost keys sampled); `bp7_s0`, `bp243_s0`; `dust7_s0`, `dust38_p4k`, `dust243_p4k` | `--preset 7m` / `--preset 243m --chain-tokens 128 --key-chunk 4`; Dust at a 4k population: `--scale 0.25` |
| Section 4.6: TinyStories | `bcts_s0`, `bpts_s0`, `dustts_s0` | the 38M commands with `--data-dir data/tinystories` |
| Section 4.6: plain GPT | `plain_bc38_s0`, `plain_bp38_s0` | `B analysis/plain_gpt.py --preset 38m --test-file test.pt`; `B analysis/backprop_preset.py --arch plain --test-file test.pt` |
| Section 5, Fig. 4: mechanism (1 GPU) | `mech38`, on the checkpoints of `bc38_s1` and `bc38_sw0` (`--save-every 488`) | `python analysis/clean_probe.py` |
| Section 3.5 / 6: FLOPs (1 GPU) | `flops38b`, `flops243` | `python analysis/flops.py --preset 38m --methods backprop,boost,cut,dust` |
| Appendix B: autograd checks (1 GPU) | `test_bc38`, `test_bc7`, `test_plain3`, … | `python analysis/test_boostcut.py --preset P [--arch plain] [--ckpt runs/<run>/final.pt]` |
| Track 1 log | `bc38f_s0` | `submission/boostcut_track1_log/`; re-scored with `uv run utils/eval.py runs/bc38f_s0/final.pt` |

## Verifying

`ledger/` holds the ledgers: every paper number tied to its runs, every run tied to its command, data, checkpoint hash,
code version and billed cost. `ledger/README.md` explains the checks.

```bash
python3 ledger/build.py --refresh   # rebuild runs.json, jobs.csv, compute.md from the records
python3 ledger/verify.py                                     # recompute every printed number; exits 1 on a mismatch
```

## Then

```bash
python3 collect.py                            # fetches results/<run>/ and prints every table
uv run --with matplotlib python llm-slop-review/figures.py
uv run --with scipy python llm-slop-review/stats.py
```

Wall-clock varies between runs on identical code and settings (seed 0 of the final configuration took 394 s once and
456 s another time, on different Modal nodes). FLOP counts are the stable comparison.
