# B64 speedup — 2026-10-07

**171.19 → 50.54 ms/update, 3.39× faster (70.48% less wall time).**

This is a paired same-host reference/optimized/optimized/reference comparison of true batch-64 Adam updates. Each pass contains 128 measured updates, after 32 warm-up updates from the same M255 parent. The historical workload is the first M256 rolling window from the immutable M256–M260 catalog: 6889 games and 203081 positions. The existing external efficiency monitor runs at its default one-second interval in every pass. The ordinary durable mark/heartbeat and concurrent 10-second pulse remain enabled on ext4.

| Pass | Mean ms/update | p50 | p95 | Updates |
|---|---:|---:|---:|---:|
| reference-1 | 176.356 | 175.649 | 227.759 | 128 |
| optimized-1 | 49.848 | 45.906 | 65.552 | 128 |
| optimized-2 | 51.225 | 48.059 | 68.656 | 128 |
| reference-2 | 166.031 | 167.919 | 228.429 | 128 |

The code changes:

- Validate Adam clocks/shapes in the same parameter order. Pack read-only moments for finite/negative checks and transfer three boolean results together. Both pre/post checks remain. On any invalid structure/value, the original detailed validator preserves first-error order and exception messages.
- Check all model weights with one packed finite predicate, after every update.
- Transfer the four returned losses and grad norm together, preserving exact Python scalar values.
- Cache cumulative position offsets once per read-only replay window. RNG seed, 64 draws, bisect and selected game/position order are unchanged. The window shares original target references and does not copy or prepack replay tensors. Mutable caller-owned sequences retain the uncached sampler.

Adam, its 84 named singleton groups and unequal inherited clocks, FP32, losses including discarded metrics, LR 2.5e-5, clip 8 with error_if_nonfinite=True, model and update order are unchanged. No compile, fused Adam, CUDA Graphs, gradient accumulation, async H2D, notification or heartbeat policy changes were introduced.

Exact preservation:

- Eight-update sample identity and telemetry comparison passes. Model and complete Adam state use exact tensor-byte hashes, including signed zero, dtype, shape, device and stride.
- Warmed states match after 32 updates. All four 128-update pass endpoints and all per-update telemetry hashes match exactly. End update counter: 55360.
- Source checkpoint/replay/manifest SHA-256 identities match before and after. No source artifact was copied, moved or changed. No production checkpoint, self-play, Arena or lineage was created.
- Synthetic isolated tests cover sampling, mutable caller behavior, missing/invalid clocks, NaN/Inf/negative moments, shape failures and original first-error ordering.

CUDA evidence (independent 16-update Nsight pass):

- Kernels/update: **3428 → 1050**.
- Explicit CUDA waits/blocking copies: **600 → 11 per update**.
- Each Adam validation: **999 → 15 kernels**, **252 → 1 wait**.
- Weight validation: **417 → 6 kernels**, **84 → 1 wait**.
- Adam remains **588 kernels/update**; its execution implementation/options were not changed.
- Nsight wall: 69.67 ms/update, +37.9% overhead. Kernel busy time: 9.682 ms/update. Nsight times are not used for the headline speedup.

Reference kernel/wait counts come from the preceding audit of this same model/optimizer path; counts are structural, while wall/kernel timings vary with clocks and profiling overhead. The earlier ~2.9× zero-host estimate assumed the old kernel workload and profile hardware state. Packing checks removes most validation kernels, so that estimate is not an absolute bound for this implementation.

Hardware sample means (device-wide; short optimized phases have few samples):

| Pass | GPU utilization % | Power W | SM MHz | Samples |
|---|---:|---:|---:|---:|
| reference-1 | 30.95 | 19.75 | 808.68 | 22 |
| optimized-1 | 31.50 | 23.62 | 968.67 | 6 |
| optimized-2 | 31.00 | 24.75 | 982.50 | 6 |
| reference-2 | 28.57 | 20.82 | 903.71 | 21 |

The two reference passes drift by several percent; both optimized passes are near 50 ms. GPU clocks/power remain dynamic and background desktop/application processes were retained. The paired speedup is measured in this environment, not a fixed-clock universal guarantee. A projected 2560-update optimizer loop falls from 7.30 to 2.16 minutes. No full 2560-update generation was launched for this comparison.

All passes report the same CUDA allocator peak in this comparison. Absolute peaks include both warmed reference/optimized trainers retained in memory and are not a single-trainer VRAM budget.

Reproduction: use the authorized V2 `b64-perf-audit` command with `configs/diagnostics/b64-speedup-comparison-20261007.json`, a fresh exclusive output directory and a read-only trainer source from audit commit `aa99db656058d4763d24bafbb7d629b5495871c1`. Its SHA is pinned in that config; create the reference checkout at the configured path or update the path while retaining the SHA. The reference is imported as a distinct class, without replacing methods or copying a training loop. Both paths run under the same signed V2 child permit. Machine-readable pass stats, source identities, byte hashes and Nsight evidence are in `report.json`.
