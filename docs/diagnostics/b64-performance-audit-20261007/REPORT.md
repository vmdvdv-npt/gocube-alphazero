# B64 training update performance audit — 2026-10-07

Normal loop: **188.10 ms/update** (two 128-update passes with the existing external monitor).
Instrumented loop: **204.47 ms/update**; overhead **+8.7%**.
PyTorch CPU+CUDA-runtime trace: **308.74 ms/update**, overhead **+64.1%** (8 updates).

M255 parent, first historical M256 rolling window from the immutable M256–M260 offline catalog; 6889 train games, 203081 positions, 32 warm-up updates. All comparison passes start from the same in-memory warmed model/Adam state. B64, LR 2.5e-5, clip 8, one real Adam update per iteration, FP32, unchanged sampler and operation order. 84 one-parameter Adam groups, 177609 parameters. No torch.compile, fused optimizer, accumulation, async H2D or optimization was introduced.

## Breakdown

CPU wall includes dispatch and waits. CUDA below is actual kernel/transfer duration from the independent 16-update Nsight pass, not CPU enqueue time or CUDA Event stream elapsed time. Percentages use the instrumented denominator so disjoint rows close correctly; profiler inflation must not be interpreted as removable overhead.

| Stage | CPU wall ms/update | CUDA kernel ms/update | p95 wall ms | % end-to-end | sync calls/update |
|---|---:|---:|---:|---:|---:|
| pre validate_clocks | 50.297 | 5.787 | 73.856 | 24.6% | 252 |
| post validate_clocks | 48.684 | 5.494 | 65.902 | 23.8% | 252 |
| batch sampling | 8.120 | 0.000 | 8.557 | 4.0% | 0 |
| torch.stack | 1.277 | 0.000 | 1.616 | 0.6% | 0 |
| H2D | 1.002 | 0.059 | 1.306 | 0.5% | 5 |
| forward | 4.739 | 16.160 | 8.271 | 2.3% | 0 |
| losses | 0.476 | 0.494 | 0.942 | 0.2% | 0 |
| unused metrics | 0.430 | 0.089 | 0.922 | 0.2% | 0 |
| loss finite check | 10.017 | 0.018 | 10.858 | 4.9% | 1 |
| backward | 11.154 | 30.228 | 22.414 | 5.5% | 0 |
| grad clip | 19.103 | 0.138 | 21.543 | 9.3% | 1 |
| Adam | 14.667 | 5.202 | 15.772 | 7.2% | 0 |
| weight finite check | 16.368 | 2.190 | 21.519 | 8.0% | 84 |
| scalar/telemetry sync | 0.277 | 0.000 | 0.444 | 0.1% | 5 |
| metrics.append | 0.002 | 0.000 | 0.003 | 0.0% | 0 |
| heartbeat/fsync | 13.778 | 0.000 | 22.054 | 6.7% | 0 |
| other Python/control | 0.880 | 0.017 | 1.149 | 0.4% | 0 |
| unattributed / residual | 3.202 | — | 4.418 | 1.6% | — |

## Bottlenecks

1. pre validate_clocks: 50.30 ms/update.
2. post validate_clocks: 48.68 ms/update.
3. grad clip: 19.10 ms/update.
4. weight finite check: 16.37 ms/update.

The gradient-clip wall includes waiting for queued backward work; its 19.10 ms is not a standalone removable clipping overhead. Heartbeat (13.78 ms) and Adam (14.67 ms) are the next largest measured stages.

Substage wall timings (inclusive; do not add these again to the table):

| Span | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| execution_permit | 0.181 | 0.171 | 0.239 |
| validate.pre.step | 0.398 | 0.385 | 0.498 |
| validate.pre.exp_avg | 17.592 | 16.426 | 26.293 |
| validate.pre.exp_avg_sq | 16.669 | 15.476 | 25.983 |
| validate.pre.negative | 11.597 | 10.947 | 15.826 |
| model.train | 0.227 | 0.218 | 0.294 |
| zero_grad | 0.379 | 0.366 | 0.443 |
| h2d.observation | 0.246 | 0.259 | 0.339 |
| h2d.pi | 0.493 | 0.455 | 0.656 |
| h2d.z | 0.090 | 0.079 | 0.165 |
| h2d.ownership | 0.099 | 0.085 | 0.195 |
| h2d.score | 0.073 | 0.064 | 0.120 |
| loss.policy | 0.172 | 0.145 | 0.340 |
| loss.wdl | 0.124 | 0.099 | 0.259 |
| loss.ownership | 0.094 | 0.078 | 0.183 |
| loss.score | 0.085 | 0.074 | 0.143 |
| metric.brier | 0.159 | 0.132 | 0.332 |
| metric.score_mae_points | 0.126 | 0.105 | 0.285 |
| metric.policy_entropy | 0.145 | 0.114 | 0.329 |
| loss.sum | 0.094 | 0.071 | 0.218 |
| loss.finite | 10.017 | 10.022 | 10.858 |
| validate.post.step | 0.370 | 0.359 | 0.448 |
| validate.post.exp_avg | 16.779 | 16.121 | 22.912 |
| validate.post.exp_avg_sq | 16.226 | 15.509 | 22.879 |
| validate.post.negative | 11.310 | 10.949 | 14.594 |
| scalar.telemetry | 0.277 | 0.257 | 0.444 |
| io.json | 0.045 | 0.042 | 0.052 |
| io.open | 0.102 | 0.092 | 0.144 |
| io.write_flush | 0.028 | 0.026 | 0.044 |
| io.file_fsync | 8.010 | 5.529 | 16.035 |
| io.replace | 0.114 | 0.113 | 0.160 |
| io.directory_fsync | 4.913 | 4.762 | 5.323 |

Adam launches 588 kernels/update across 84 singleton groups (5.202 ms GPU). Gradient clipping launches 14 kernels/update (0.138 ms GPU), including norm reduction/nonfinite check/scaling, with one CPU wait. The clip trace contains one reduce_kernel, one lpnorm_cleanup, two multi_tensor_apply kernels, one batched-copy kernel and nine vectorized elementwise kernels per update. The recorded foreach/fused options remain their inherited defaults (None); PyTorch 2.4.1 already auto-selects foreach internals for the CUDA singleton Adam groups. The audit has not enabled a new optimizer implementation. A future optimizer experiment must compare explicit execution/grouping choices to this actual default, rather than assume the baseline is a pure scalar Adam loop.


Validation Python loop/control is the difference between each inclusive validate span and its four subspans (step, exp_avg, exp_avg_sq, negative variance); it also includes profiling span setup/teardown. Adam step tensors are CPU scalars in this checkpoint; their int conversion does not synchronize CUDA.

## GPU and synchronization

Nsight loop: 253.04 ms/update (+34.5% vs baseline). 3428 kernels/update, 65.82 ms kernel busy/update, 26.0% of traced loop wall. Median kernel duration 5.28 µs; p95 GPU idle gap 0.168 ms. This is timeline busy fraction, distinct from sampled SM utilization.

The installed PyTorch 2.4.1/CUPTI trace produces CUDA runtime calls but no device activity (also reproduced with a single multiply). Missing activity is **unavailable**, never zero GPU work. Nsight Systems 2026.1 records the device activity successfully without changing training/runtime dependencies. NVIDIA documents WSL2 tracing support for this architecture in [CUPTI special configurations](https://docs.nvidia.com/cupti/13.1.1/special-configurations/special-configurations.html).

Actual waits are attributed to CUDA runtime synchronize/copy calls inside spans: Python boolean conversion of finite/all/any checks in both validations and weight checks; loss boolean check; nonfinite gradient check in clip_grad_norm_; four float(loss.detach()) and float(grad); and blocking .to(cuda) transfers. The isfinite/all/any kernels themselves enqueue asynchronously; scalar consumption forces the wait. clip_grad_norm_ reduction and scaling kernels remain separate from its nonfinite-check wait, which can also absorb queued backward work. The loss finite check can absorb completion of preceding forward/loss/metric work, so its wall time cannot all be removed by changing that check.

Sampled normal-loop device statistics:

- utilization.gpu: mean 36.23, p95 42.00, 47 samples.
- power.draw: mean 19.15, p95 19.51, 47 samples.
- memory.used: mean 1671.00, p95 1671.00, 47 samples.
- temperature.gpu: mean 49.96, p95 51.00, 47 samples.
- clocks.sm: mean 638.96, p95 810.00, 47 samples.

## Batch, H2D and I/O

H2D payload: 167168 bytes/update: observation=103680, pi=20992, z=768, ownership=41472, score=256. All five .to(cuda) calls retain their original blocking behavior. Stack and sampling are measured independently; CUDA transfer duration and CPU blocking/enqueue wall are different quantities.

One training mark performs one durable write and two fsync calls: file plus directory. Thus the 2560 B64 training marks guarantee **2560 writes and 5120 fsyncs**, plus the concurrent 10-second pulse and lifecycle marks. At baseline duration, approximately 48.2 pulse writes are expected during training; this is a projection, not an observed whole-generation count. A full generation was not run. Measured bounded-pass counts (exclude initial heartbeat):

- normal-monitor-1: 130 writes, 260 fsyncs, 2 pulse writes.
- normal-no-monitor-1: 130 writes, 260 fsyncs, 2 pulse writes.
- normal-no-monitor-2: 130 writes, 260 fsyncs, 2 pulse writes.
- normal-monitor-2: 130 writes, 260 fsyncs, 2 pulse writes.
- instrumented: 130 writes, 260 fsyncs, 2 pulse writes.
- profiler: 8 writes, 16 fsyncs, 0 pulse writes.

Mean durable main-thread heartbeat: 13.567 ms. JSON serialization, exclusive file creation, write/flush, file fsync, replace and directory fsync are shown in substage timings. Both diagnostic and production heartbeat files use ext4; /tmp is tmpfs and was excluded from the final I/O comparison.

## External monitoring and limitations

ABBA means: with monitor 188.10 ms/update; without monitor 185.65 ms/update (+1.3% difference). This is a bounded comparison, not a causal confidence interval; thermal/power drift and desktop GPU workloads remain possible. The existing monitor runs at its default one-second interval, reads diagnostic heartbeats/process telemetry and invokes nvidia-smi. Only audit-owned monitor PIDs are terminated. No production telemetry is stopped or reconfigured.

The initial exploratory pass overlapped local CPU tests and is excluded from the final report. Final measured passes run after those tests completed. No production trainer/monitor process was found at preflight; desktop device memory/utilization is still device-wide. Other active Python processes belong to codex_bot and codex_telegram_events.py; these host/application processes were retained and their marginal influence was not isolated. Warm-up changes the model by 32 genuine updates; all benchmark comparisons reuse that same state, and no state is saved. CUDA Events are read only after each entire measured pass, with no per-stage synchronization; their stream elapsed values include idle/dispatch gaps and are retained in JSON as stream_ms. Nsight labels resolve actual device work through CUPTI correlation IDs. Autograd worker-thread launches inherit the enclosing main-thread backward range in the same process; no stage is inferred from GPU execution timestamp alone.

## Opportunity and next PR

Amdahl scenarios (assumptions, **not measured speedups**): science-neutral lower-risk ~1.44× if half the validation/weight-check and batch wall is saved; more invasive execution ~2.18× if 80% of these plus heartbeat wall is saved. Loose theoretical zero-host/zero-I/O bound: 2.9×.

| Next change | Expected gain hypothesis | Scientific/numerical risk | Bitwise trajectory | Complexity |
|---|---|---|---|---|
| Consolidate existing validation booleans into fewer CPU transfers while retaining every check | Save 30–60 ms/update in the instrumented regime; benchmark separately | Low numerical risk; failure reporting/order needs review | Plausible; require exact parity | Medium |
| Consolidate weight finite and telemetry scalar transfers | Hypothesis: save 5–10 ms/update | Low; preserve fail-closed behavior and returned values | Plausible; require exact parity | Medium |
| Cache cumulative replay offsets for an immutable window; retain RNG and bisect order | Hypothesis: save 4–6 ms/update | Low if replay identity invalidates cache | Expected exact | Low–medium |
| Review durable heartbeat cadence/batching in a separate explicitly authorized policy PR | Hypothesis: save 5–10 ms/update, ceiling ~14 ms | Training math unaffected; supervision/durability risk | Training state can remain exact | Medium |
| Evaluate Adam group coalescing or explicit optimizer execution only after host checks are addressed | Hypothesis: save 3–8 ms/update; current ~15 ms wall is the ceiling, automatic foreach already present | Reduction/order and optimizer rounding may change | Not guaranteed | Medium–high |

Discarded brier/score_mae_points/policy_entropy remain computed. Removing them would require a separate change; their measured dispatch/kernel cost is visible above and cannot explain the large fixed update cost. The Amdahl upper bound also ignores irreducible memory transfers and execution dependencies and should not be interpreted as an attainable forecast.

## Preservation and reproduction

Four-update CUDA parity: sampled positions, model tensors, complete Adam state and update counter all exactly equal. CPU parity is covered by isolated synthetic-replay tests. All registered source checkpoint/replay/manifest SHA-256 identities match before and after the audit. No checkpoint/replay was copied, changed, moved, archived or published; no self-play or Arena was created. Production notifications, services, live working tree and code pins were untouched.

Run `.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint b64-perf-audit configs/diagnostics/b64-perf-audit-20261007.json` from this branch, selecting a new exclusive output directory on the production filesystem. Source reference paths, SHA-256 identities, measurement code hashes, environment flags, all pass statistics and Nsight summary are in `report.json`. Raw profiler/monitor artifacts remain in the diagnostic output directories outside the lineage.
