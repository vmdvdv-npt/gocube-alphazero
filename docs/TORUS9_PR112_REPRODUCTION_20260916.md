# TORUS9 PR112 controlled reproduction — 2026-09-16

Scope: one controlled standard-64 Golden iteration on current `main` after PR112. No optimization sweep, Arena run, workload change, or scientific-setting change was performed.

## Before / after

Delta is `PR112 after - PR110 before`; negative is faster for wall-clock metrics.

| Metric | PR110 before | PR112 after | Delta | Delta % |
|---|---:|---:|---:|---:|
| Full iteration until publication (s) | 273.738240520 | 324.644318178 | +50.906077658 | +18.596626% |
| Self-play wall (s) | 187.048342982 | 222.962663612 | +35.914320630 | +19.200555% |
| Self-play serialization (s) | 17.007436742 | 26.166984566 | +9.159547824 | +53.856133% |
| Sample build (s) | 28.824037887 | 39.548907114 | +10.724869227 | +37.208074% |
| Validation / stamping (s) | 10.300861562 | 0.009669025 | -10.291192537 | -99.906134% |
| Replay update / validation / fingerprint (s) | 17.064829642 | 18.475424093 | +1.410594451 | +8.266092% |
| Training wall (s) | 5.859997693 | 6.439437859 | +0.579440166 | +9.888061% |
| Checkpoint serialization / verification (s) | 2.401372892 | 2.522382669 | +0.121009777 | +5.039191% |
| Moves/s | 25.298272760 | 21.223284308 | -4.074988451 | -16.107773% |

## Run identity and Golden contract

The after run used the exact PR110 Phase A final run identity so derived game and training seeds remained unchanged:

- Before: `runs/torus9/archive/torus9-nightly-20260916-phase-a-final/`
- After: `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/`
- After commit: `0c41727619b3ba3d06d802b4ee735f5f52511d5a` (PR112 merge)
- Profile fingerprint: `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`
- Rules fingerprint: `sha256:e0fd15c82d42a63ca05c3b6fb3ae02deb938543e1e06483ecfeab275dc98a39e`
- Target fingerprint: `sha256:02ab244688534b271473302ab4edf00516b91b8c9e592d2a8de63dfb5`
- Self-play contract fingerprint: `sha256:22a4e4dd37d70bd3d712b909476120b96385802ec874b99358fab256c4e3351f`

The original Google Sheet `Golden Standart`, tab `TORUS 9×9`, was read directly. The run matched the required Golden: 64 games, 64 MCTS simulations, 16 workers × 4 active games, 64 contexts, cap64/wait1ms, fast search OFF, root noise ON, komi 0.5, Adam, LR 0.001, batch64, 80 optimizer updates, 5120 samples, rolling-3 replay with cap20,000, and gating OFF/decoupled. No `.xlsx` was created or used, and Golden parameters were not changed.

## Targeted PR112 measurement

The requested calculation is:

```text
validation_saved_sec = 10.300861562005593 - 0.009669025006587617
                     = 10.291192536999 s
```

The `sample_validation_and_stamping` bucket now contains the remaining stamping work; the two generic duplicate full validation passes are no longer present in this path. This is a real targeted improvement, not a relocation into `replay_update_and_validation`: that neighboring bucket changed from `17.064829642 s` to `18.475424093 s`, a `+1.410594451 s` change.

The full-iteration calculation is:

```text
iteration_saved_sec = 273.7382405200042 - 324.6443181779905
                    = -50.906077657986 s
iteration_speedup_pct = -18.596626310%
throughput_ratio = 273.7382405200042 / 324.6443181779905
                 = 0.843194306
```

Thus the targeted bucket is confirmed, but the one controlled after run did not produce an end-to-end wall-clock win. The observed neighboring increases were self-play `+35.914 s`, serialization `+9.160 s`, sample build `+10.725 s`, replay update `+1.411 s`, training `+0.579 s`, and checkpoint serialization `+0.121 s`. The raw evidence does not support attributing all of those changes causally to PR112; they are recorded as observed run-to-run performance, not silently normalized away.

## Raw self-play telemetry

The after run recorded:

- 64 requested / 64 completed / 64 valid games; technical games `0`.
- 4,732 total moves; `21.223284308` moves/s; `0.287043575` games/s; `1,033.356869` games/h.
- Startup `0.256682946 s`; teardown `1.534724278 s`; postprocessing/accounting `8.488400746 s`; serialization `26.166984566 s`.
- 16 worker processes, 16 PIDs, zero restarts, zero worker failures.
- Effective process-tree CPU `5.450516806` cores; worker CPU `943.918727579 s`; process-tree CPU `1,163.085820403 s`.
- Inference rows `299,679`; telemetry `total_rows = 299,679`; forward calls `17,675`; no dropped or duplicated inference rows.
- Batch mean / p50 / p95 / max: `16.954965 / 17 / 38 / 59` rows.
- Worker→broker latency mean `0.533091 ms`; broker queue mean `5.373523 ms`; H2D mean `0.148654 ms`; model forward mean `5.491357 ms`; response→worker mean `1.008476 ms`.
- Blocked inference calls `98,467`; blocked inference mean `17.912350 ms`, p50 `17.190592 ms`, p95 `26.663616 ms`; summed blocked wait `1,763.775406 s`.
- Pending games final `0`; configured active-context target `64`; peak search contexts `64`; the existing occupancy telemetry remains event-count based.

## Raw replay / training telemetry

- Sample build: `39.548907114 s`.
- Validation and stamping: `0.009669025 s`.
- Replay update, replay validation, and fingerprint: `18.475424093 s`.
- Training wall: `6.439437859 s`.
- Training H2D/batch construction `0.221432232 s`; forward `0.527265120 s`; loss `0.062167174 s`; backward `1.374167314 s`; optimizer `0.194275632 s`; parameter accounting `0.183539962 s`.
- Fresh/replay rows: `4,732`; source generation `[1]`; unique replay row IDs `4,732`.
- Adam step `0 → 80`; optimizer updates `80`; samples consumed `5,120`; unique sampled rows `3,099`; batch size `64`.
- Checkpoint serialization and verification `2.522382669 s`; checkpoint publication `0.000168301 s`.
- Checkpoint was saved and re-verified successfully. Model hash: `sha256:d6406daf9dfc7d3f97fdfd1f5cb4c9680d9307d827ce69d1e6c984ee37037478`.

The normalized self-play payload fingerprint (`77d2b94094e8c926ac802fcf8caed23e77329cd79a3297de247b364519fe159e`) and normalized replay payload fingerprint (`7b4dd07071aa91b0232d2d4a30ec30ba4c28fb0202db463da2c4be77ef1a1ffb`) match the PR110 final artifact. The deterministic sampled-row fingerprint is `sha256:5fe524ff319c2a380f60befd3a56a450186397ac479af90d3298deec4093f353`.

## Correctness gates

All correctness gates passed for the after run:

- 64/64 games completed; technical games `0`.
- Inference row accounting is exact; worker failures and restarts are empty/zero.
- Replay row count is `4,732`; deterministic replay row IDs are unique and correct.
- Optimizer updates are `80`; samples consumed are `5,120`; Adam clock is continuous from step `0` to `80`.
- Checkpoint save and repeat verification succeeded.
- Komi is exactly `0.5`; profile, rules, target, self-play, architecture, optimizer, replay, and model fingerprints remain on the Golden contract.
- No model/search/training semantics or execution preset was changed.

Occupancy caveat retained from PR110: `mean_active_contexts = 32.25196850393701` is the event-count average `4096 / 127`, not time-weighted occupancy. `tail_duration_after_pending_empty_sec = 0` does not establish that no drain tail existed. Exact time-weighted occupancy and drain-tail duration are not measured by current telemetry and were not changed in this reproduction.

## Next measured bottlenecks — no fixes applied

1. Remaining sample build: `39.548907114 s` after versus `28.824037887 s` before. Its absolute removal ceiling is the measured stage itself; the realistic ceiling is lower because conversion and target construction are required.
2. Replay update / validation / fingerprint: `18.475424093 s` after versus `17.064829642 s` before. The measured stage is the upper bound; the removable validation portion is not separately instrumented.
3. Self-play serialization: `26.166984566 s` after versus `17.007436742 s` before. Full elimination is only a theoretical ceiling because encoding and persistence remain required.
4. Postprocessing overlap: the after postprocessing, serialization, and sample-build stages sum to a `74.204292426 s` overlap envelope, but CPU/memory contention and transaction ordering make the realizable saving materially smaller and unmeasured.
5. CPU/search ↔ broker synchronization: blocked mean `17.912350 ms` and broker queue mean `5.373523 ms`; aggregate concurrent wait cannot be converted into a defensible wall-clock ceiling without new causal instrumentation.

## Verdict

**PR112 NOT CONFIRMED** for the required end-to-end wall-clock pass: the duplicate validation removal is directly confirmed in its target bucket (`10.291193 s` saved), correctness and scientific fingerprints are preserved, but this strict same-seed after run is `50.906078 s` slower overall and has a throughput ratio of `0.843194306`. No follow-up optimization was implemented.

## Raw artifact references

Before raw artifacts:

- `runs/torus9/archive/torus9-nightly-20260916-phase-a-final/phase-a-report.json`
- `runs/torus9/archive/torus9-nightly-20260916-phase-a-final/timing/M01.json`

After raw artifacts:

- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/manifest.json` — SHA-256 `7c05986e509b176ab5bc64cf5daa48e9fbec1524a948ae5f82f7dd3af4549c4c`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/phase-a-report.json` — SHA-256 `5ad1cc70231c00792f70e074d1750dc82db5cbb62601137c1f4ad79c7613210e`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/timing/M01.json` — SHA-256 `b073b2303145b4bcd39c48bfbd82d78068e2b8f8138e5265767d1260f454b427`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/selfplay/M01-games.jsonl` — SHA-256 `7e496e5f9c8ae57ba861373137c934874b637489cd9e23889650ec67f94db4ea`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/replay/iter-01-fresh.jsonl` — SHA-256 `2b2615ee09545f165d0842a472b95b209ad68ab761360118d54454018a36f418`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/checkpoints/M1.metadata.json` — SHA-256 `5c9958f4d598fb9d5403f672131d042d6e8218703ee32c30773cab6cee0ce65a`
- `runs/torus9/archive/torus9-pr112-repro-20260916-exact3/checkpoints/M1.pt` — SHA-256 `b2001d956642c79e40977308b0550c710d618bfb48c1813c430a4da901e5f986`

PR110 interpretation reference: `docs/TORUS9_PR110_PIPELINE_FORENSICS_20260916.md`.
