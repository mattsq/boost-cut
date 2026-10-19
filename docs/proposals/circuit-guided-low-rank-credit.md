# Proposal: Circuit-guided low-rank credit assignment for forward-only pretraining

**Status:** Implemented as experimental code ([`noise/train/cgsc/`](../../noise/train/cgsc/README.md)); no GPU experiments run yet, so Gates A–C are open
**Target:** `spicylemonade/boost-cut`, primarily `noise/train/boostcut/run.py`
**Date:** 2026-10-19
**Working name:** Circuit-Guided Subspace Credit (CGSC)

## Executive summary

Boost → Cut demonstrates that token-local perturbation experiments can produce useful forward-only learning signals, and that following the strongest MLP routes can reduce the cost of propagating those perturbations. This proposal tests a distinct, potentially complementary economy: **restrict the directions in which residual-stream credit is measured to a compact, dynamically refreshed subspace**.

It then asks whether these same subspaces can constrain the weight update, first as a projected-gradient control and subsequently through LoRA-like temporary low-rank adapters whose changes are periodically merged into the base model. The intended combination has three separable restrictions:

1. **Computational-route sparsity:** Cut's existing top-`route_k` MLP pathway approximation.
2. **Credit-direction sparsity:** evaluate loss perturbations along `r` carefully chosen directions instead of a full `d`-direction Hadamard cycle.
3. **Update-rank restriction:** apply a rank-`r` or rank-`r_u` update to selected projection matrices, with periodic merges to permit cumulative high-rank learning.

**Primary hypothesis:** forward-observable activity and route statistics identify subspaces capturing enough of the true *useful credit* that the reduction in measurement cost exceeds the optimization penalty, especially after Boost has organized strong routes.

**Crucial negative possibility:** the directions explaining the current forward pass may differ from the directions required to improve the network. Sparse activations alone do not imply low-rank gradients. This proposal makes that distinction the first experiment and includes explicit failure criteria.

Do **not** begin by implementing LoRA and sparse probing together. First establish that forward-derived subspaces capture useful gradient directions, then show a wall-clock gain from restricted probes with full-rank parameter updates, and only then introduce low-rank parameterization.

## 1. Context and exact integration points

The repository's benchmark implementation is `noise/train/boostcut/run.py`; the locked challenge entry is `entries/track1_boostcut/run.py`. Preserve the latter unchanged until a candidate has passed dedicated ablations and verification.

Relevant current mechanisms:

- `PRESETS` gives widths 128, 256, 512, and 1024 for the 2M/7M/38M/243M model ladder; `HIDDEN = 4 * WIDTH`.
- `Boost.basis()` returns a sign-flipped full Hadamard basis of width `WIDTH`.
- `Boost.probe()` measures local readout credit. Its default full-cycle path uses `cycle_derivatives(...)` and `derivative @ Hs / WIDTH`; a separate `--boost-probes` option implements partial random Hadamard sampling.
- `Cut.chain_credits()` uses one full residual-stream basis cycle per token/site and `chain_chunk(...)` to measure end-to-end, **attention-cut** credit.
- `Cut.estimate()` selects MLP routes with `top = (relu(hpre)^2 * ||c_proj[:, unit]||).topk(route_k)`. In the 38M configuration `route_k` defaults to 128 of 2048 units. Those routes are a *downstream perturbation-evaluation approximation*, not an existing low-rank update constraint.
- `Capture.reassemble()` converts a measured module output-error tensor into weight gradients by outer products with captured inputs. The optimizer is ANVIL II on block matrices and Adam for other parameters. The full gradient construction and distributed aggregation occur in `Boost.step()`.
- The present Boost → Cut handoff occurs at `--switch`, default near 30% of training; any proposal must preserve initialization, model architecture, token order, optimizer treatment, and evaluation policy in the first controlled comparison.

See the repository [README](https://github.com/spicylemonade/boost-cut/blob/main/README.md), [run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py), [paper draft](https://github.com/spicylemonade/boost-cut/blob/main/llm-slop-review/draft.md), and [reproduction notes](https://github.com/spicylemonade/boost-cut/blob/main/llm-slop-review/README.md).

### What this is not

- This is **not** the claim that top active neurons automatically have the largest gradients.
- This is **not** standard LoRA fine-tuning of a frozen pretrained backbone. In pretraining, the whole model must acquire new circuits; a permanently fixed low-rank adapter can prevent that.
- This is **not** a way to eliminate all full-width operations or attention computations. Its first target is the number/cost of **perturbation evaluations**, not baseline forward-pass FLOPs.
- This is **not** an unbiased estimator of the unrestricted gradient if the measurement subspace is fixed and incomplete. It is a projected gradient for a specified surrogate objective, subject to Cut's additional attention and route approximations.

## 2. Mathematical formulation

Let a site-specific residual vector be `h ∈ R^d`, and let `F_s(h)` denote the loss actually measured at site `s`: the shared-head local loss in Boost, or Cut's attention-frozen, route-restricted downstream loss in Cut. Its local target credit is

`g_s = ∇_h F_s(h)`.

Choose an orthonormal basis `Q_s ∈ R^{d×r}`, `Q_s^T Q_s = I_r`, with `r << d`, and define the projector `P_s = Q_s Q_s^T`. Perturb only in directions `q_j` from this basis:

`d_j = [F_s(h + ε q_j) - F_s(h)] / ε`,  `ĝ_s = Σ_{j=1}^r d_j q_j`.

For small `ε`, `ĝ_s ≈ P_s g_s` up to finite-difference error. With symmetric differences, this error can be reduced at the cost of twice as many evaluations. Direction-probe count falls from `d` to `r`, **but end-to-end time need not fall by `d/r`**, since baseline forward capture, token batching, vocabulary softmax, attention-internal probes, optimizer work, and kernel efficiency may dominate.

For compatibility with the existing Hadamard convention, set `B = sqrt(d) Q_s`, so columns of `B` have squared norm `d`; then reconstruction is `(derivatives @ B^T) / d`. This is **not** the same as the existing random-partial-Hadamard estimator's `1/K` normalization. The latter estimates the full gradient in expectation over randomly chosen directions. The proposed deterministic-subspace estimate deliberately targets `P_s g_s` and must use consistent scaling. Unit-vector probes with `Σ_j d_j q_j` are an equivalent, and clearer, reference implementation.

For a linear module `y = Wx`, the projected output error gives a weight gradient

`Ĝ_W = (P_s g_y) x^T`.

For a batch this is a sum of outer products, so `rank(Ĝ_W)` is at most `r` when every sample uses the **same** output projector `P_s`. If bases vary per token, the batch gradient need not be rank `r`. Therefore the first low-rank design should use a shared basis per site over a refresh interval, rather than different bases for every token.

### Connection to LoRA

Represent selected weights during an interval as

`W = W_base + Q_s A`, with `A ∈ R^{r×d_in}` and `Q_s` fixed and orthonormal.

Then `∇_A L = Q_s^T ∇_W L`, which can be assembled from the same projected activation credit without computing a dense unrestricted gradient. At refresh, merge `Q_s A` into `W_base`, zero/reinitialize `A`, choose a new `Q_s`, and continue. Over multiple intervals the **cumulative** change can have higher rank than each temporary adapter.

This is a fixed-left-factor version of LoRA, chosen for compatibility with output-credit measurements. Later compare with conventional two-factor LoRA (`B A`) and learnable bases. Updating `Q` itself requires off-subspace information; it must not be assumed available from `Q`-restricted probes.

**Optimizer caveat:** applying full-matrix ANVIL II or decoupled weight decay to a projected dense gradient does not necessarily keep the resulting parameter displacement in the chosen subspace. Treat *projected credit*, *projected parameter update*, and *actual low-rank adapter training* as separate experimental conditions. Explicitly measure post-optimizer leakage `||(I-P)ΔW||/||ΔW||`.

## 3. Central questions and falsifiable hypotheses

**H1. Alignment.** A subspace derived from the model's forward activity or strongest MLP routes captures substantially more held-out activation-gradient energy than a random subspace of the same rank.

**H2. Predictive utility.** Alignment with the current gradient predicts one-step loss reduction on unseen minibatches. Gradient-energy capture alone is insufficient when curvature and step size matter.

**H3. Learnability.** Replacing full basis cycles with circuit-informed restricted cycles reduces wall-clock and FLOPs to a fixed validation-loss target without a compensating quality loss larger than a prespecified tolerance.

**H4. Dynamic basis.** Periodic basis refresh beats a permanently fixed basis at equal total probe cost. A small fraction of exploration measurements off the currently selected basis prevents early lock-in.

**H5. Parameter efficiency.** After H1–H4 succeed, a fixed-left-factor low-rank update plus merge preserves most of the benefit of unrestricted projected updates while reducing gradient/optimizer memory and potentially additional execution time.

The null is genuinely plausible: activation PCs and route-aligned directions may explain forward variance yet capture little optimization-relevant credit, or subspace construction and synchronization may outweigh saved probe work.

## 4. Stage 0: diagnostic study before training modifications

### 4.1 Measurements

Use existing checkpoints from initialization, early Boost, just before handoff, just after handoff, middle Cut, and late training. Read a fixed held-out diagnostic token sample across at least three seeds. Record per site/layer:

- Clean residual streams, inputs to `attn.c_proj` and `mlp.c_proj`, hidden preactivations, activation `relu(hpre)^2`, output-column norms, and selected route indices.
- Site-specific full-basis credit for the *exact* local Boost loss and *exact* attention-cut, route-restricted Cut surrogate, separately.
- Diagnostic true gradients from autograd for the full model loss, on small batches, explicitly marked **oracle for analysis only**, not used to train candidate methods.
- Gradient energy by singular direction and by layer; principal angles and basis overlap across steps/seeds.
- Projection scores on **held-out diagnostic examples**, not the same tokens used to estimate the basis.

Use `noise/analysis/test_boostcut.py` as an implementation reference for checking credit against autograd; keep true gradient calls isolated to diagnostic scripts.

### 4.2 Candidate subspaces

For identical `r ∈ {8,16,32,64,128,256}` (where legal for preset width), compare:

1. **Random orthogonal:** fixed or seeded Haar/Hadamard subspace. This controls for rank alone.
2. **Activation PCA:** top `r` eigenvectors of a streamed covariance of site activations, estimated on earlier training batches.
3. **Contribution-weighted directions:** for an MLP projection, aggregate outer products of output columns weighted by observed token-level contribution `relu(hpre)^2 * ||c_proj[:, j]||`. Use its top eigendirections in residual space; do not merely select an arbitrary `r` of 2048 scalar neurons and call them an orthonormal residual basis.
4. **Route covariance:** aggregate the actual output contributions `v_j = relu(hpre_j)^2 c_proj[:, j]` across tokens and selected routes; compute the leading eigenvectors of `Σ v_j v_j^T` (possibly normalized by token counts). This captures coherent mixtures of routes rather than raw activation magnitude.
5. **Gradient-SVD oracle:** leading left singular vectors of historical true site credit, for diagnostic upper-bound comparison only.
6. **Time-shuffled or layer-shuffled basis:** falsification controls for spurious covariance structure.

Rank is fixed by construction; distinguish PCA of representations (`h`), PCA of contribution vectors, and PCA of *credit* (`g`). The last is not forward-only unless estimated with explicit exploration probes.

### 4.3 Scores

For held-out credit vectors `g_i`, report `ρ = Σ ||Q^T g_i||² / Σ ||g_i||²`, cosine between `g_i` and `P g_i`, and signed directional agreement. Calculate these against **both** the surrogate credit being optimized and the full-loss gradient; they are not interchangeable.

Also evaluate the loss after a tiny normalized update for selected layer parameters, on different held-out tokens. Compare to an equal-norm random projected update, a projected exact-gradient update, and an unprojected exact-gradient update. Track evidence of gradient components outside the strong-activation subspace, especially at initialization.

**Gate A (go/no-go):** at r ≤ d/8, route or activity bases must demonstrate repeatable held-out alignment above random, preferably capturing ≥80% of the *surrogate* credit energy across important sites after Boost. Treat 80% as an initial engineering target, not a theoretical requirement. If no cheap forward-only basis beats random, stop the circuit-derived component and report the negative result.

## 5. Stage 1: subspace credit without LoRA

### 5.1 Minimal implementation

Introduce `SubspaceBank` in a **new experimental module**, e.g. `noise/train/boostcut/subspace.py`, not by rewriting the benchmark entry. Responsibilities:

- Maintain per-site (and optionally phase-specific) orthonormal `Q` in float32, along with acquisition-step metadata, rank and refresh generation.
- Construct bases from streaming activation/route statistics; use thin SVD or symmetric eigendecomposition on at most `d×d` covariance matrices. Share/reduce statistics across ranks deterministically.
- Support `random`, `activation_pca`, `route_cov`, `oracle_gradient_svd` (diagnostics only), and `none` (full basis).
- Orthonormalize and verify `||Q^TQ-I||`; carry deterministic sign conventions and checkpoint basis state.
- Hold basis **fixed during the optimizer step**; update only between steps. Never select the basis based on the loss responses of the same probes without accounting for selection bias.

Refactor `Boost.probe()` into a generic site-credit interface accepting a `(d, r)` orthonormal basis and an explicit reconstruction convention. Refactor `Cut.chain_credits()` so that the move axis has `r` probe directions instead of assuming exactly `WIDTH`; scalar central-difference moves must remain appended and indexed correctly. `chain_chunk()` already accepts a variable-length move axis `(m, P, WIDTH)` and is the natural route through the existing compiled kernels.

Do not change `Cut` route selection, attention-frozen semantics, head credit, `hidden_credit()`, query/key/value/gate probes, model, or optimizer in this first stage. For any site whose result is not yet safely projected, use its original full-cycle estimator. The stage-1 purpose is to isolate **residual-stream** measurement cost.

### 5.2 Practical probe details

- Distinguish unit-normalized `q_j` from Hadamard rows with norm `sqrt(d)` and compare actual logit perturbation magnitude. Tune `epsilon` to maintain comparable finite-difference signal-to-noise without leaving the locally linear regime; report a step-size sweep.
- Do not silently use `1/r` normalization for `Q`-restricted credits. Projections are deterministic and biased relative to full credit by design; report the projection residual.
- Always preserve a full-basis path (identical to original code) as a regression test and a periodically sampled oracle.
- Start with shared bases per block site and phase. Token-specific top-k route masks may remain token-specific; they are conceptually separate.
- Estimate memory and arithmetic costs before implementation: reductions in probe count need not reduce cached activation size or dense output-gradient/optimizer allocations.
- Test compiled kernel shapes for r=16, 32, 64, 128 and `WIDTH`; cache compiled variants or bucket ranks to avoid compilation noise.

### 5.3 Exploration and adaptive refresh

Restricted probes contain no information about orthogonal components. Reserve optional **off-subspace sentinel probes** for diagnosing missing signal. Options, in increasing complexity:

1. A small seeded random orthogonal complement sample every `T` steps. Compute loss-response energy for these directions but do not update from them in the simplest experiment.
2. Refresh based on route/activation covariance every `T` steps, independent of probe responses.
3. Use the sentinel estimates to augment the basis with directions carrying unexpectedly large loss response, then reorthogonalize and drop low-value basis directions.

Log off-subspace response energy and basis turnover. Compare `refresh_T ∈ {never, 32, 128, 512}` optimizer steps, subject to enough data for stable covariance. Make **exploration probe cost and basis construction cost count toward training time**.

### 5.4 Baselines and acceptance

Run the official 38M setup under identical data order and token budget, 3–5 seeds:

- Original Boost → Cut, `route_k=128`, full-width credit.
- Random rank-r credit with unchanged optimizer.
- Activation-PCA rank-r credit.
- Route-covariance rank-r credit.
- Gradient-SVD rank-r **oracle upper bound**, not an eligible forward-only training result.
- Each of the above with and without 5–10% effective probe budget reserved for off-subspace exploration.
- For any promising case, a factorial comparison with `route_k ∈ {64,128,256,2048}` to test whether route sparsity and directional sparsity save independent work.

**Gate B:** demonstrate a reproducible lower *total* wall-clock (and independently measured FLOPs) to fixed held-out validation targets, with statistical uncertainty. Suggested initial threshold: ≥1.25× speedup at validation loss no worse than +0.02 relative to the matched baseline after equal tokens, plus a meaningful time-to-target gain. This is a **decision threshold**, not a predicted outcome. Full-quality training loss and test loss must also be reported to detect validation-target gaming.

## 6. Stage 2: restrict parameter updates with fixed-left-factor adapters

Only after Gate B succeeds, add a temporary adapter for **selected** projection weights:

`W_eff = W_base + Q A`  (rank at most r).

Start with `transformer.h.{i}.attn.c_proj.weight` and `transformer.h.{i}.mlp.c_proj.weight` because their outputs share the width-`d` residual space whose basis has already been measured. Keep the full base matrices in the forward pass. Initially continue unrestricted ordinary updates for embeddings, output head, mixing scalars, and attention-internal / MLP-input matrices so early learning is not bottlenecked at every parameter group. Later expand cautiously to `c_fc`, q/k/v/gates with appropriately defined **output-space** bases (hidden width differs from residual width).

For each selected matrix during an interval:

1. Maintain a frozen `W_base`, a fixed site-specific orthonormal `Q`, and trainable `A` initialized to zero.
2. Use measured `ĝ_y` to obtain `G_A = Σ (Q^T ĝ_{y,i}) x_i^T` directly. No autograd is required.
3. Update `A` using an explicitly specified optimizer (begin with AdamW). Do **not** pretend ANVIL II on `W` is equivalent to AdamW on `A`; perform a matched optimizer control and tune learning rate on the same budget.
4. Every `M` optimizer steps, merge `Q A` into `W_base`, reset adapter and its optimizer state by a prespecified rule, refresh `Q`, and continue. Record merge overhead, accumulated effective-rank statistics and forgetting from resets.
5. Restore checkpoints exactly, including `Q`, `A`, `W_base`, optimizer state, RNG, training phase, basis acquisition step and dataset position.

Controls:

- Projected-credit / **unrestricted dense parameter update** (stage-1 method).
- Dense projected step `ΔW = -η Q Q^T G_W` under plain SGD, which exactly enforces the subspace, to separate geometry from optimizer differences.
- Fixed-left-factor `Q A`, no merge.
- Fixed-left-factor `Q A`, periodic merge.
- Conventional LoRA `B A` under matching rank/optimizer, trained from available measured module output credit; distinguish two-factor optimizer effects from the geometry of `Q`.
- Random-basis adapter and gradient-oracle-basis adapter (diagnostic bound).
- An unrestricted original Boost → Cut run and a standard backprop run with matched architecture/data, for absolute context.

**Gate C:** comparable held-out quality/time-to-target with lower trainable-parameter optimizer state, and evidence that merged updates can accumulate meaningful rank. If adapter memory falls but time-to-target worsens, report it as a memory-saving method rather than a training speedup.

## 7. Stage 3: adaptive subspace discovery and pretraining dynamics

Only if Stage 2 is useful, investigate:

- **Warm-start effect:** is the activity/route basis more informative after Boost than at random initialization? Compare immediate Cut, Boost-only, and handoff at fractions 0.1, 0.3, 0.5.
- **Basis drift:** principal angles between successive `Q` and relation to learning-rate schedule, layer depth, and route turnover. Do stronger routes become stable while off-subspace gradient mass diminishes?
- **Emergence of new circuits:** temporarily loosen route sparsity or increase sentinel probes; measure whether unexplored directions predict later-useful pathways.
- **Adaptive rank:** grow r when off-subspace response energy or validation degradation rises; shrink it when capture saturates. Include probe and update cost in a single controller objective.
- **Unfrozen attention effects:** if cross-token effects are restored in later Cut variants, do subspaces learned on attention-cut surrogates still capture gradients of the improved estimator? Do not assume transfer.
- **Scaling:** 7M smoke/diagnostic runs; 38M multi-seed studies; 243M single-seed feasibility followed by replications if promising; longer token budgets and a second dataset to detect benchmark specialization.

## 8. Experiments, metrics, and reporting

### 8.1 Model and data policy

Honor the Noise benchmark's fixed architecture, data and training accounting when making benchmark claims. Experimental work lives in new files under `noise/analysis/` and a new training entry; do not edit the registered official entry in `entries/` before verification. Use the pinned data and checksum approach from `ledger/` for reproducibility. Record code SHA, model preset, seeds, environment, accelerator type/count, CUDA/PyTorch versions, compiled kernel status, and optimizer/schedule.

### 8.2 Metrics required for every training comparison

- Held-out validation and test loss, **plus loss versus wall-clock, tokens, optimizer steps, and FLOPs**, including warmup/compilation separately and transparently.
- Time to agreed thresholds (e.g. 4.2 and an additional tighter quality target), and final loss under fixed tokens and fixed GPU-hours.
- Cost decomposition: clean forward, Boost readout probes, Cut chain probes, basis statistics/eigendecompositions, attention-internal estimates, gradient reconstruction, optimizer, distributed communication, merges, and evaluation.
- Peak allocated GPU memory, total optimizer state, checkpoint size, synchronization bytes, and achieved hardware utilization.
- Surrogate projected-credit alignment, full-loss gradient alignment, basis rank/turnover, off-subspace probe energy, route overlap and leakage after optimizer application.
- Per-seed numbers and confidence intervals. Evaluate multiple seeds at promising settings; do not promote a single best seed.

Do not multiply a theoretical reduction `d/r` by Cut's reported route cost reduction to claim an overall speedup. Profile the combined system, including fixed costs and interactions. Where practical, report both end-to-end and isolated probe-kernel costs.

### 8.3 Ablation matrix, staged to limit expense

| Question | Comparison | Primary readout |
|---|---|---|
| Does circuit structure predict useful credit? | Random vs activity PCA vs route covariance vs gradient oracle | Held-out `ρ`, one-step loss decrease |
| Is rank the relevant bottleneck? | r = 8,16,32,64,128,256,full | Quality vs probe time |
| Is selected basis actually better than random? | Equal-r basis, equal refresh cost | Time-to-target |
| Does the basis need to evolve? | No refresh vs fixed-T vs response-triggered refresh | Quality, turnover, costs |
| Is off-subspace learning necessary? | 0%, 5%, 10% sentinel exploration | Quality and missing response energy |
| Are routes and directions complementary? | `route_k` crossed with r | Interaction and FLOPs |
| Does rank-restricted updating help? | Projected dense SGD/ANVIL, `Q A`, merged `Q A`, conventional LoRA | Quality, memory, time |
| Does this survive beyond short runs? | 38M full budget, 243M, longer token horizon | Sustained loss gap, scalability |

## 9. Correctness tests

Implement automated tests (small dimensions on CPU where possible; GPU integration where necessary):

1. **Projector mathematics:** `Q^T Q ≈ I`, `P² ≈ P`, and deterministic unit basis exact recovery of a known linear loss's projected gradient.
2. **Hadamard compatibility:** with `r=d`, the new measurement path agrees with the existing full-cycle estimator under exactly matched direction norm and perturbation magnitude. Include both central- and forward-difference tests.
3. **Normalization test:** with `r<d`, the result equals `P g`, **not** `(d/r) P g` (the latter is an unbiased random-subset expectation convention, not a fixed-subspace projection).
4. **Surrogate autograd:** full-basis Cut chain credit matches autograd for the *attention-frozen/route-restricted surrogate*, and projected Cut credit matches `P` times that surrogate gradient within finite-difference tolerance. Separately measure disagreement with the full-network gradient.
5. **Attention invariance:** perturbation credit does not inadvertently reenable downstream token mixing in the Cut approximation.
6. **Writer gradient:** for a toy linear layer, `G_A = Q^T G_W`, reconstructed full gradient `Q G_A` equals `P G_W`, including minibatch aggregation and DDP reductions.
7. **Merged adapter:** output is invariant (within precision) to `W_base ← W_base + Q A`, `A ← 0`, with appropriate cache/optimizer handling; checkpoint resume gives identical continuation.
8. **No leakage:** assert that forward-only production training uses no `.backward()` / `torch.autograd.grad` or hidden autograd graph; diagnostic/oracle scripts may use autograd and must be visibly separated.
9. **Regression:** with feature flags disabled, loss curves, speed and outputs reproduce the original entry to normal numerical tolerance.
10. **Accounting:** reported wall-clock includes basis refresh and exploration; untimed compilation is reported explicitly; official ledger/checkpoint validation continues to pass.

## 10. Suggested code architecture and CLI

Proposed additions (illustrative, not yet implemented):

```text
noise/
  train/boostcut/
    run.py                     # retained as reference; minimal guarded hooks only
    subspace.py                # SubspaceBank and Q builders
    projected_credit.py        # shared Boost/Cut probe interface and tests
    adapters.py                # optional fixed-left-factor adapters/merge
  analysis/
    subspace_alignment.py      # Stage 0 diagnostics incl oracle gradients
    subspace_costs.py          # FLOP/time accounting
    test_subspace_credit.py    # mathematical and integration tests
  train/cgsc/
    run.py                     # experimental entry; original is unchanged
```

Suggested flags:

```text
--credit-basis {full,random,activation-pca,route-cov,oracle-grad}
--credit-rank {8,16,32,64,128,256}
--basis-refresh-steps INT
--basis-warmup-steps INT
--exploration-probe-fraction FLOAT
--subspace-target {boost,cut,both}
--update-mode {dense,projected-sgd,qa,qa-merge,lora}
--adapter-rank INT
--adapter-merge-steps INT
--route-k INT                  # existing option, retained
```

`oracle-grad` must be flagged diagnostic-only and excluded from forward-only claims. The fixed no-feature path should remain bitwise or tolerance-equivalent to the original wherever possible.

### Phased work packages

**PR 1: Diagnostic evidence.** `subspace_alignment.py` + tests and a short results report. No training changes. Result: decide whether forward-derived bases beat random and how alignment evolves.

**PR 2: Basis-restricted credit.** Add `SubspaceBank`, flexible probe shapes and correctness checks. Keep dense parameter updates, same route selection and optimizer. Result: cost/quality frontier and Gate B.

**PR 3: Dynamic refresh and exploration.** Off-subspace sentinel budget, deterministic refresh and checkpoint support. Result: fixed vs dynamic basis, exploration controls.

**PR 4: Adapter training.** Fixed-left-factor `Q A`, merge, optimizer matching, memory/time analysis. Result: Gate C.

**PR 5: Scaling and reproducibility.** Multiple seeds, alternate dataset, 243M, longer horizon, verify ledger; prepare a benchmark entry only if the method legitimately improves the official objective.

## 11. Failure modes and interpretations

| Failure | Detection | Response |
|---|---|---|
| Forward activation variance misidentifies learning directions | PCA `ρ` at/near random; poor one-step gain | Stop PCA approach; evaluate route-covariance or abandon circuit basis |
| Strong-route selection locks out emerging features | Off-subspace responses rise; early quality gap | Increase exploration, route_k, early full probing, or refresh frequency |
| Token-varying bases destroy update-rank guarantee | Batch gradient singular rank >> r | Share Q by site over an update window |
| Low-rank update handicaps pretraining | Equal-token loss gap grows after many merges | Merge more often, expand rank, or retain dense writers |
| Restricted probing reduces theoretical arithmetic but not time | End-to-end profiler shows overhead/poor utilization | Bucket probe ranks, fuse kernels, or reject speed claim |
| Bias from Cut surrogate dominates effects | Good surrogate alignment but low full-gradient alignment | Keep separate interpretation; test restored-attention estimator |
| Dynamic basis is selected on current probe noise | Held-out alignment collapses | Lag basis construction; independent diagnostic batches |
| ANVIL/Adam transforms projected gradients off-subspace | Update leakage is large | Test explicit projected SGD and adapter updates separately |
| Apparent improvement results from mismatched compute/optimizer | Reproducibility audit fails | Match tokens, seeds, learning-rate sweeps, costs and hardware |

A negative result in H1 is still scientifically informative: sparse *forward* routes need not induce compact *backward* credit directions. A positive Stage 0 with negative Stage 1 suggests the implementation's fixed costs dominate. A positive Stage 1 with negative Stage 2 argues for basis-guided credit measurement **without** rank-restricting updates.

## 12. References

- Nick Cammarata, [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (2026): motivating view of concentrated and legible computation; not evidence by itself of gradient low rank.
- [Boost → Cut repository](https://github.com/spicylemonade/boost-cut): baseline implementation, measurements, validation, ledger.
- [Q Labs Noise benchmark](https://github.com/qlabs-eng/noise): fixed forward-only pretraining challenge.
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685): low-rank update factorization.
- Lialin et al., [*ReLoRA: High-Rank Training Through Low-Rank Updates*](https://arxiv.org/abs/2307.05695): periodically merged low-rank updates during training.
- Zhao et al., [*GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*](https://arxiv.org/abs/2403.03507): adaptive low-rank optimizer representation for pretraining.
- Evci et al., [*Rigging the Lottery: Making All Tickets Winners*](https://arxiv.org/abs/1911.11134): dynamic sparse connectivity; reminder that useful sparse structure changes during training.

## 13. Decision summary

**Do Stage 0 first.** The key research question is not whether LoRA can be attached to Boost → Cut; mechanically, it can. The key is whether a *cheaply inferred, evolving, forward-observable circuit basis* captures the *credit directions needed for learning*. If it does, restricted perturbation measurement may be useful on its own; if low-rank adapters also work, the two savings can compound. If it does not, combining the restrictions is likely to make training less capable while obscuring why.