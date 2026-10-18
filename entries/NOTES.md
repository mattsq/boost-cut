# Notes on the entries

Each entry changes one track script in place, as the benchmark asks, and adds its run's `config.json`, `metrics.jsonl` and `result.json` as its log.

---

## Track 2: Boost, stagewise forward-only training (4.123 in 314 s)

**Changes `train/speedrun/run.py`** (`submission/boost_speedrun_run.py`). Boost replaces Dust's end-to-end credit
with a stagewise objective: every attention and every MLP is trained only to lower the loss of the shared head
reading the stream right after it. A readout loss is a per-token function of one stream, so each token's credit is
measured exactly by perturbing that stream. Forward differences along a full sign-flipped order-512 Hadamard cycle
are each evaluated as the actual soft-capped cross-entropy of the head, normalized by the moved stream's own RMS.
There are no reruns of later blocks and no cross-token interference. The writers use that credit directly. MLP
hidden units, queries, keys, values, gates and value embeddings take one local step from it. Keys, values and gates
use a matched-filter kernel from the clean attention. The model, init, optimizer (ANVIL II plus Adam), rates and
schedule shape are the record's. Batch 4: the two GPUs of each sequence split its probes and draws.

**Result:** val 4.1230 in 314.2 s on 8×H100, seed 0, `reached_target: true` (logs in `submission/boost_track2_log/`).
The current record is 4.184 in 1,239 s.

*Rule notes for reviewers:*
- The intermediate readouts are a training objective (the logit lens). The final model and its evaluation are
  unchanged.
- Judgement calls, disclosed:
  - the matched filter weights each query's score by the clean attention pattern;
  - the embedding's stage reruns the first attention with the other positions' keys and values held clean;
  - value embeddings reuse the value draws (a value move is a value-embedding move scaled by the gate).

---

## Track 1: Boost → Cut, exact credit along strong routes (3.945 in 456 s)

**Changes `train/val/run.py`** (`submission/val_run_boostcut.py`, byte-identical to `noise/train/boostcut/run.py`
as run; only its default `--output` would become `logs/val`). Two phases over the one epoch. The model, init,
optimizer (ANVIL II plus Adam), rates and schedule shape are the record's.

1. **Boost, the first 30% of the epoch** (the Track 2 Boost entry, plus keys from single-key experiments). Every
   attention and every MLP learns to lower the loss of the shared head reading the stream right after it. Each token's
   readout credit is measured exactly by forward differences along a full sign-flipped Hadamard cycle of its stream.
2. **Cut, the remaining 70%.** End-to-end credit with every attention output held at its clean value, so a move of one
   token's stream reaches only that token's loss and every token is its own experiment. A full cycle of moves at the
   raw embedding and at every block's mid stream runs forward through all later blocks to the token's own loss. Inside
   each later MLP the move travels only through the token's top 128 of 2,048 units by output contribution; the others
   are held at their clean outputs. Block outputs reuse the same moves (a move of block d's output is a move of block
   d+1's mid stream times its residual scalar). The mixing scalars are measured by their own ±h moves through the same
   chain. Writers, the MLP hidden layer, queries, values, gates and value embeddings take one local step from that
   credit, as in the record. Keys use single-key experiments: moving one key changes exactly one logit per query, so
   all T × d_head experiments per head are evaluated in closed form from the clean probabilities and scored against
   each query's target. That is the construction the record already uses for its head (one logit at a time).
3. **Handoff.** Boost runs at 4 sequences per update (two GPUs per sequence), Cut at 8 (one per GPU). Weights, ANVIL's
   rails and Adam's moments carry over, and the data order is the benchmark's.

**Result:** val 3.9448 (re-scored with `utils/eval.py`: 3.94481) in 456.4 s on 8×H100, seed 0 (logs in
`submission/boostcut_track1_log/`). The record is 3.991 in 72 min. Seeds 1 and 2: val 3.9335 and 3.9595 (mean of
three 3.946 ± 0.013), in 456 s and 401 s.

*Rule notes for reviewers:*
- Every learning signal is a change of a loss caused by a perturbation, measured by running the network forward,
  either at the site it trains or one local step from such a measurement. Nothing is carried through weights,
  attention or skip connections by formula or by finite differences beyond that step. Every measured credit agrees
  with autograd of its estimand (`noise/analysis/test_boostcut.py`, cosine ≥ 0.996 at every site).
- Judgement calls, disclosed:
  - Cut holds every attention output at its clean value during a move (its estimand is the attention-cut gradient);
  - Cut holds each token's weak MLP units (outside its top 128) at their clean outputs;
  - the attention internals are scored against `c_proj`'s view of the measured mid-stream credit (equal to scoring
    their moves through `c_proj`);
  - the values' matched filter uses the clean attention pattern, like the record's decay kernel;
  - Boost's intermediate readouts are a training objective (the logit lens); the final model and its evaluation are
    unchanged.
