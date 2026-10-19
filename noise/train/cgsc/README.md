# Circuit-guided subspace credit (CGSC)

The code for [the proposal](../../../docs/proposals/circuit-guided-low-rank-credit.md). It is experimental: the
benchmark script `train/boostcut/run.py` and the entries in `entries/` are unchanged. With its defaults
(`--credit-basis full --update-mode dense`), `train/cgsc/run.py` runs the same training as `train/boostcut/run.py`,
through the original code path. `analysis/test_subspace_credit.py` checks that the estimates are then identical, bit
for bit.

| file | role |
|---|---|
| `train/boostcut/subspace.py` | `SubspaceBank`: per-site orthonormal bases (`random`, `activation-pca`, `contrib`, `route-cov`, `oracle-grad`), statistics, refresh, sentinels, checkpoint state |
| `train/boostcut/projected_credit.py` | Boost and Cut subclasses that probe active sites along their basis (Boost readouts, Cut chain, top stream) |
| `train/boostcut/adapters.py` | writer updates: `projected-sgd`, `qa`, `qa-merge`, `lora`; update-leakage measurement |
| `train/cgsc/run.py` | the training entry and its flags |
| `analysis/subspace_alignment.py` | stage 0: alignment of each basis with the Cut, full-loss and Boost credit (autograd oracle), one-step test, Gate A summary |
| `analysis/subspace_costs.py` | FLOPs and vocabulary evaluations per token at each rank, statistics and refresh cost, compiled timings |
| `analysis/test_subspace_credit.py` | the proposal's correctness tests (section 9), on CPU or GPU |

## Sites and what is restricted

A site is a residual-stream position:
- `emb`: the raw embedding;
- `mid{i}`: block i's stream after its attention;
- `out{i}`: block i's output.

Boost probes `mid{i}` and `out{i}`. Cut probes `emb`, every `mid{i}`, and the top stream `out{L-1}`. Only these
residual-stream probes are restricted. Boost's hub probes (in the attention output's space) keep the full cycle. The
following are all unchanged:
- route selection and frozen attention;
- `hidden_credit()`;
- queries, keys, values and gates;
- the mixing scalars;
- the head.

The basis statistics are always taken from the clean forward pass:
- `contrib` and `route-cov` use the MLP of the site's own block (`emb` uses block 0's);
- `activation-pca` uses the site's own stream.

Both phases share one bank, so Cut starts from bases built during Boost.

The writer updates (stage 2) use each writer's output site:
- `attn.c_proj` of block i: `mid{i}`;
- `mlp.c_proj` of block i: `out{i}` in Boost. In Cut it is `mid{i+1}` (where Cut measures that credit), or
  `out{L-1}` for the last block.

The handoff merges the adapters and restarts them on Cut's sites.

## Running the stages

**Stage 0 (no training changes).** Save checkpoints along a baseline run with the existing flag. At 38M,
`--save-every 488` saves every 10% of the epoch, and the handoff falls at 1,464 sequences. Then score them:

```bash
torchrun --standalone --nproc_per_node=8 train/boostcut/run.py --seed 0 --save-every 488 --output logs/bc_s0
python analysis/subspace_alignment.py --preset 38m --output logs/align_s0.jsonl \
    --ckpts init,logs/bc_s0/ckpt_488.pt,logs/bc_s0/ckpt_1464.pt,logs/bc_s0/ckpt_1952.pt,logs/bc_s0/ckpt_2928.pt,logs/bc_s0/final.pt
```

Repeat for seeds 1 and 2. The `gate_a` lines at the end of the output compare each forward-only basis's mean `rho`
on Cut's estimand at r ≤ W/8 with a random basis of the same rank, and give the share of sites at `rho ≥ 0.8`.
**If no forward-only basis beats random, stop here** (Gate A).

**Stage 1 (restricted probes, dense updates).** These runs are the Gate B comparisons, with the same data order and
token budget, over 3 to 5 seeds:

```bash
for basis in random activation-pca route-cov oracle-grad; do
  torchrun --standalone --nproc_per_node=8 train/cgsc/run.py --credit-basis $basis --credit-rank 64 \
      --basis-refresh-steps 128 --eval-every 488 --test-file test.pt --profile-every 50 --leakage-every 50 \
      --output logs/cgsc_${basis}64_s0
done
# exploration: --exploration-probe-fraction 0.1 --exploration-use {measure,augment,update}
# (augment finds a missing direction from a few random sentinels in a ~450-dimensional complement per step, so it
#  needs long refresh windows: hundreds of steps, not tens)
# factorial with routes: --route-k {64,128,256,2048}
```

**Stage 2 (only after Gate B).** Add `--update-mode`:
- `projected-sgd`, with `--adapter-lr`;
- `qa`;
- `qa-merge`, with `--adapter-merge-steps`;
- `lora`.

Each writer moves only by `W_base + Q A`, and its AdamW state covers the factors alone. Compare these runs against
the stage-1 dense run at the same basis and rank.

## Reading the logs

Each `metrics.jsonl` record can carry these extra fields:

| field | meaning |
|---|---|
| `cgsc.off_{site}` | the sentinels' estimate of the share of credit energy outside the basis |
| `cgsc.turnover_{site}` | 1 minus the overlap with the previous basis, at each refresh |
| `cgsc.generation` | how many refreshes have run |
| `cgsc.update_leakage` | \|(I − P) ΔW\| / \|ΔW\| of the writers' actual steps, P from the bank's current basis (under `qa` modes it is non-zero between a refresh and the next merge, when the adapter still holds the old Q) |
| `cgsc.cumulative_effective_rank` | the effective rank of each writer's total change since the adapters engaged, at each merge |
| `profile` | the time split of a synchronized step |

`train_seconds` includes the basis statistics, refreshes and exploration probes. It excludes the leakage and
effective-rank diagnostics, which are logged separately as `diag_seconds`. Runs with `oracle-grad` are marked
`diagnostic_only` in `config.json` and `result.json`. They measure the full cycle, so they are bounds, not
forward-only results.

## Not implemented

- Resuming a run from its checkpoint. `train/boostcut/run.py` has no resume either. The bank and the writers save and
  load their full state (`cgsc_<tag>.pt` next to each checkpoint, tested in `test_subspace_credit.py`), but the
  entry does not restore a run.
- Bases for `c_fc`, queries, keys, values or gates (stage 2's later expansion).
- An adaptive-rank controller (stage 3).
