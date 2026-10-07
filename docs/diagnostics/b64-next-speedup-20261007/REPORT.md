# B64 Adam execution batching — 2026-10-07

**53.68 → 38.76 ms/update: 27.79% less wall time, 1.38× speedup.** Historical current-optimized baseline: **50.54 ms/update**. Fresh paired current-optimized/candidate/candidate/current-optimized passes are the acceptance evidence.

M255 parent; first M256 rolling window of the immutable M256–M260 catalog; 6889 games / 203081 positions. Each trajectory starts with 32 warm-up updates and measures 128 true B64 updates. The default external efficiency monitor, durable heartbeat/marks and concurrent 10-second pulse stay enabled. Background desktop applications were retained. Tests ran separately from the final paired benchmark and Nsight captures. An earlier exploratory comparison overlapped tests and used an uncached reference sampler; it is excluded.

## Overhead attribution

Fresh, independent 16-update baseline and candidate Nsight captures. Rows are disjoint; subspans are not counted twice. Loss construction includes 14 loss kernels, 15 retained auxiliary-metric kernels, 4 sum kernels and 4 finite-check kernels. Forward is 99 kernels and backward 275. CPU sampling and zero_grad(set_to_none=True) launch no CUDA kernels.

| Component | Baseline kernels/update | Candidate kernels/update | Baseline waits/update | Candidate waits/update |
|---|---:|---:|---:|---:|
| forward/backward | 374 | 374 | 0 | 0 |
| loss construction + retained metrics + sum + finite check | 37 | 37 | 1 | 1 |
| zero_grad | 0 | 0 | 0 | 0 |
| grad norm / clip_grad_norm | 14 | 14 | 1 | 1 |
| Adam | 588 | 11 | 0 | 0 |
| Adam validation (pre + post) | 30 | 30 | 2 | 2 |
| weight validation | 6 | 6 | 1 | 1 |
| telemetry/scalar extraction | 1 | 1 | 1 | 1 |
| H2D / replay sampling | 0 | 0 | 5 | 5 |
| other | 0 | 0 | 0 | 0 |
| TOTAL | 1050 | 473 | 11 | 11 |

All 11 waits are accounted for: blocking H2D copies for observation, pi, z, ownership and score (5); pre/post packed Adam predicate transfers (2); ordinary-loss finite check (1); clip_grad_norm_ nonfinite check (1); weight finite check (1); packed losses/grad-norm telemetry (1). No wait is removed or deferred. Counts mean explicit synchronize/blocking cudaMemcpy calls, using the existing audit definition.

## Change and exactness

OrdinaryAdam inherits PyTorch Adam and preserves its param_groups/state_dict/load_state_dict. Compatible FP32 CUDA groups contribute tensors to one execution list, in original parameter order, using the same PyTorch functional Adam and automatic foreach dispatch. Names, 84 singleton groups, every group option, CPU per-parameter clocks and complete moments remain untouched. Parameters are independent; each retains the same ordered arithmetic operations and its own bias correction. The baseline already used foreach: seven kernels for each singleton group caused 84 × 7 = 588 launches. Batching the same stages needs 11 kernels (including multi-tensor chunking) across the whole optimizer.

Other devices/dtypes, differing group options, aliased parameters, sparse gradients, closures, differentiable/capturable/fused/explicit non-foreach execution and scaler attributes use the ordinary superclass dispatch. No parallel validator or custom Adam mathematics is introduced. This uses PyTorch 2.4.1+cu124 functional Adam and _init_group; the exact gate should be rerun when upgrading the runtime.

The comparison harness builds the reference module's own cached replay window where supported, so both current-optimized and candidate use cached sampling offsets. Older trainer modules keep their existing list fallback. The short gate now checks complete model/Adam byte fingerprints after every update.

- Sample games, positions and order match for all eight short-test updates.
- Every short-test update matches model bytes, complete optimizer bytes including groups/clocks, and telemetry. Hashes include dtype, shape, device, stride and signed-zero tensor bytes.
- States match exactly after 32 warm-up updates.
- All four 128-update endpoints and all per-update telemetry byte hashes match. End update: 55360.
- An isolated synthetic CUDA test checks parameter and complete Adam bytes at every one of 160 updates with unequal clocks; ordinary torch.optim.Adam loads the candidate state unchanged.
- 125 local tests pass. CUDA/CPU validator cases preserve first-error type/message for missing/invalid clocks, missing moments, NaN/Inf, negative variance, shape mismatch and multiple errors. CUDA nonfinite-gradient and post-update nonfinite-weight cases match ordinary Adam exception type/message and failure-state fingerprints.
- Source checkpoint/replay/manifest SHA-256 identities match before/after both timed and structural audits and again at report assembly. No source artifact was copied, moved or modified. No production training artifacts, self-play, replay, Arena or lineage were created.

## Paired wall time

| Pass | Mean ms/update | p50 | p95 |
|---|---:|---:|---:|
| reference-1 | 54.119 | 52.775 | 71.191 |
| optimized-1 | 39.069 | 37.324 | 58.957 |
| optimized-2 | 38.455 | 37.727 | 45.326 |
| reference-2 | 53.239 | 50.727 | 66.315 |

Both candidate passes beat both baseline passes; reduction against the paired baseline mean is 27.2% and 28.4%. Against the historical 50.54 ms/update mean, the candidate mean is 23.3% lower; that cross-session number is contextual, not a paired comparison.

| Pass | GPU utilization % | Power W | SM MHz | Samples |
|---|---:|---:|---:|---:|
| reference-1 | 34.71 | 23.92 | 789.29 | 7 |
| optimized-1 | 36.20 | 24.95 | 1004.60 | 5 |
| optimized-2 | 34.40 | 24.97 | 838.20 | 5 |
| reference-2 | 36.50 | 23.73 | 672.33 | 6 |

GPU statistics are device-wide with dynamic clocks and few samples per short pass. All timed passes report 90,528,768 bytes peak allocated and 119,537,664 bytes peak reserved. Both warmed trainers are retained; these are comparison-process peaks, not a single production trainer budget.

Independent Nsight wall means: 64.54 ms baseline and 60.41 ms candidate. Profiler timings are excluded from headline wall results. Kernel busy times: 10.469 → 7.380 ms/update.

Projected 2560-update loop: **137.42 → 99.23 seconds** (2.29 → 1.65 minutes). Historical 50.54 ms baseline projects to 129.38 seconds. No full 2560-update generation was launched.

**Scientific training configuration is unchanged:** Adam, FP32, batch 64, LR 2.5e-5, clip 8, error_if_nonfinite=True, inherited unequal clocks, losses/metrics, sample RNG/order, checkpoint format and optimizer serialization. No fused Adam, compile, CUDA Graphs, accumulation, AMP or TF32 change. Validation and notification/heartbeat policy are preserved.

## Reproduction and evidence

Use the existing V2 production_entrypoint b64-perf-audit command with configs/diagnostics/b64-next-speedup-comparison-20261007.json. Create a read-only cfed84a checkout at the reference path or update that path retaining its pinned trainer SHA. Select fresh exclusive diagnostic output directories. For Nsight, use the documented capture-range command and the candidate nsys config; run the baseline nsys config from the cfed84a checkout. Report.json contains full pass statistics, source/code identities, per-update short hashes, trajectory fingerprints, hardware statistics and both structural profiles. Raw outputs remain outside training lineage under /home/codex/diagnostics/gocube-b64-next-speedup-20261007-*.
