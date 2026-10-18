# Forward-only pretraining on the Noise benchmark

These are learning rules that train a GPT with no backward pass, entered on the
[Noise benchmark](https://github.com/qlabs-eng/noise): a fixed 37.7M-parameter GPT trained for one epoch on 10M
FineWeb tokens. Every learning signal comes from measured changes in the loss when parts of the network are nudged;
nothing is backpropagated. The [ledgers](ledger/) tie every number here to its run, code, data and cost.

## Results

| entry | track | val loss | train time, 8×H100 | record when run |
|---|---|---|---|---|
| **Boost → Cut** | 1: lowest loss | **3.945** | 456 s | 3.991 in 4,303 s |
| **Boost** | 2: time to 4.2 | 4.123 | **314 s** | 4.184 in 1,239 s |

Both are seed 0, from the official logs in `entries/`. Over 3 seeds, Boost → Cut reaches held-out test loss
3.935 ± 0.012. Backprop with the same model and optimizer reaches 3.706 ± 0.009. On our own 8×H100 nodes, the
Track 1 record's recipe reproduces at 3.992 in 4,292 s, and the speedrun record's script runs in 1,146 s, so Boost
is 3.6× faster on the same hardware.

- **Boost.** Every attention and MLP layer learns only from the shared output head reading the stream right after
  it. Each token's credit is measured exactly, by nudging its stream along a full set of orthogonal directions.
- **Boost → Cut.** Boost runs for the first 30% of training. End-to-end credit follows, with attention held fixed,
  measured only through each token's 128 strongest of 2,048 MLP units.

**Why Boost first.** A layer's readout loss depends on a single token's stream, so nudging that stream along a full
set of orthogonal directions measures the token's credit exactly, with no noise from other tokens. Every layer gets a
clean signal from the first step. Boost also leaves each token's work concentrated in a few strong MLP units, and
that makes the next phase cheap.

**Why Cut next.** Boost alone stalls at 4.062 test loss: each layer improves its own readout, not the final output.
Cut gives each layer its real effect on the final loss:
- Holding attention fixed makes every token its own experiment, so the measurement stays exact.
- Following only each token's strongest units makes it 2.4× cheaper, for 0.006 of loss.

With Boost's head start Cut reaches 3.935; without it, 3.984 (all three over 3 seeds).

## What's here

- `entries/`: each entry's script and official log. A script replaces its track's script in the benchmark
  (`train/val/run.py` or `train/speedrun/run.py`); `NOTES.md` describes each one.
- `noise/`: the benchmark repository plus this work's code: `train/boostcut/run.py` and the paper's analysis scripts.
- `llm-slop-review/`: the paper draft and its figures. An LLM wrote it and it is still under review; a few of its
  runs are still in progress.
- `results/`: the config, metrics and result of every run behind the paper.
- `ledger/`: the ledgers.
- `tools/`: the Modal launcher and queue used for every run.

## Checking it

```bash
python3 ledger/verify.py
```

This does two checks:
- It recomputes every number in the paper from `results/` and the run logs.
- It checks that every checkpoint re-scores to its logged loss.

`ledger/verification.md` is its last report. `ledger/runs.json` ties each run to its command, data, checkpoint hash,
code version and Modal app, and `ledger/compute.md` reconciles compute with Modal's billing; see
[`ledger/README.md`](ledger/README.md). The two entries' checkpoints are attached to the
[entries-v1 release](https://github.com/spicylemonade/boost-cut/releases/tag/entries-v1); their sha256 hashes match
`ledger/rescore.jsonl`, and `uv run utils/eval.py <file>` re-scores either one.

## Running an entry

```bash
cd noise && uv sync && uv run utils/prepare_data.py
cp ../entries/track1_boostcut/run.py train/val/run.py
uv run torchrun --standalone --nproc_per_node=8 train/val/run.py --output logs/val
```
