# PR110 Torus9 pipeline forensics — 2026-09-16

Scope: evidence-only analysis of the committed PR110 diagnostics plus one minimal execution fix. No workers/cap/wait sweep was run. Scientific settings remain the Golden Standard: 64 games/iteration, 16 workers × 4 active games, 64 contexts, cap64, central wait1 ms, 64 MCTS simulations, komi 0.5, 80 Adam steps, 5120 sample draws.

## 1. Startup and teardown

PR110 Phase A reports:

- self-play startup: `0.21161604698863812 s`
- self-play teardown: `0.6399634830013383 s`
- combined: `0.8515795299899764 s`
- Phase A iteration wall until publication: `273.7382405200042 s`

Startup + teardown are therefore only `0.311%` of the measured iteration. The historical long pre/post barrier delay is not present in this path and is not a useful optimization target.

## 2. Exact decomposition of the 56.190 s replay/sample-preparation bucket

The PR110 raw diagnostic contains the three sub-timers directly:

| Stage | Wall time | Share of 273.738 s iteration |
|---|---:|---:|
| sample build | `28.82403788701049 s` | `10.530%` |
| sample validation + stamping | `10.300861562005593 s` | `3.763%` |
| replay update + replay validation + fingerprint | `17.064829641996766 s` | `6.234%` |
| **total** | **`56.18972909101285 s`** | **`20.526%`** |

Code audit explains why this path is expensive. For record-driven training, the current Torus9 adapter validates every generated sample inside `build_samples`; the generic `TrainingEngine` then validated the same source samples again, validated the copied/stamped samples again, and `update_replay` validated stamped samples again before replay mutation. `validate_sample` is not a cheap shape check: it reconstructs Torus state/legal context and checks policy/visits/WDL/auxiliary target provenance.

PR110 had already disabled a further validation pass inside `train_fixed_budget`, but the orchestration-level duplicates above remained.

## 3. What the 17.007 s self-play serialization bucket contains

Measured bucket: `17.007436741987476 s` (`6.213%` of the Phase A iteration).

The PR110 timer wraps one expression:

```python
write_jsonl(games_path, [record.to_dict() for record in records])
```

The code path contains all of the following work inside that single timer:

1. `Torus9SelfPlayGameRecord.to_dict()` calls `dataclasses.asdict`, recursively copying nested game/position structures.
2. `to_dict()` then applies `_jsonable`, performing another recursive normalization pass.
3. `write_jsonl` applies `_jsonable` to every already-converted row again.
4. `json.dumps(..., sort_keys=True)` encodes every full game record.
5. The legacy writer joins the encoded rows into one large string.
6. `write_text` performs the filesystem write.

PR110 did not instrument these six pieces separately, so the existing artifact cannot truthfully assign seconds to conversion vs encoding vs disk I/O. `17.007 s` is exact; any finer time split would require a new instrumented reproduction and must not be fabricated.

## 4. 64-context occupancy and drain-tail audit

For the 64-game Golden run, configured games and context capacity are both exactly 64. All 64 games are placed into worker queues before execution, so there is no reserve queue to replenish a finished context.

The observable occupancy event sequence is therefore:

```text
startup events:    1, 2, 3, ..., 63, 64
completion events: 63, 62, 61, ..., 2, 1
```

PR110 reports `mean_active_contexts = 32.25196850393701`. That value is exactly:

```text
(sum(1..64) + sum(1..63)) / 127 = 4096 / 127 = 32.25196850393701
```

This proves that the metric is an event-count average, not a time-weighted occupancy measurement. It contains no information about how many wall-clock seconds were spent at 64, 32, 8, or 1 active contexts.

The existing `tail_duration_after_pending_empty_sec = 0.0` is also not a real zero drain tail. In `SelfPlayEngine`, `pending_empty_at` is only set when a non-empty reserve queue becomes empty during replenishment. With 64 games and 64 initial contexts, the reserve queue is already empty before execution (`pending_started_nonempty = False`), so the timer is never armed.

Therefore an exact time-domain occupancy curve and exact drain-tail duration cannot be reconstructed from the committed PR110 artifact. The structural drain is real; its wall duration was not recorded. No number is invented here.

## 5. Savings envelope from existing evidence

All figures below distinguish measured stage time from ceilings/proxies.

### Repeated replay/sample passes

Measured total replay/sample-preparation wall: `56.190 s`.

The entire 56.190 s is not removable: sample construction, one semantic validation boundary, replay mutation, replay integrity validation, and fingerprinting are required. The two post-build measured buckets (`10.301 + 17.065 = 27.366 s`) are the upper envelope containing repeated validation plus required stamping/replay/fingerprint work.

The fix in this PR removes two generic duplicate full validation passes for record-built samples while preserving adapter validation at sample construction and immediately before replay mutation. PR110 did not time individual validation loops, so the exact wall saving must be measured in a later single reproduction; it is not claimed from static analysis.

### Self-play game-record serialization

Measured: `17.007 s`.

- 50% reduction would save `8.504 s`, reducing the measured 273.738 s iteration to about `265.235 s` (`~3.2%` throughput improvement).
- Complete elimination is only a ceiling: `17.007 s` saved, about `256.731 s` total (`~6.6%` throughput improvement).

The stage is a valid later target, but existing evidence does not identify which internal serialization component dominates, so it is not the first code change in this PR.

### Overlap of postprocessing with still-running self-play

Potentially overlap-friendly measured work is at most:

- self-play postprocessing: `5.207 s`
- game-record serialization: `17.007 s`
- sample build: `28.824 s`

Total theoretical envelope: `51.038 s` (`18.64%` of the measured iteration).

This is only an upper bound. Real overlap would contend for CPU/memory bandwidth with MCTS and central inference, and deterministic ordering/transaction semantics must be preserved. It is not justified as a first change without the missing time-domain drain telemetry.

### CPU/search ↔ broker synchronization

PR110 Phase A reports:

- `98,467` blocked inference calls
- total blocked inference: `1563.279 s` summed across workers
- blocked mean: `15.876 ms`
- broker queue mean: `4.886 ms`
- model forward mean: `4.592 ms`
- self-play wall: `187.048 s`

Broker queue is about `30.8%` of mean blocked latency. Multiplying that ratio by self-play wall gives `57.6 s`, but this is **not** a valid wall-clock saving prediction because worker waits overlap. It is only a latency-component proxy. The current artifact cannot convert aggregate concurrent wait into an exact iteration saving.

## 6. Selected fix

Selected first cause: **duplicate generic validation passes in the replay/sample preparation path**.

Reason:

- it is directly visible in the measured 56.190 s critical path;
- the redundancy is proven by code, not inferred from utilization;
- both current production adapters (Torus9 and Cube) validate rows produced by `build_samples`;
- both validate stamped rows in `update_replay` before replay mutation;
- removing the two intervening generic passes preserves fail-closed semantic boundaries, replay contents, fingerprints, optimizer work, model science, and artifact transaction semantics;
- it requires no execution-parameter sweep and does not alter Golden scientific settings.

The engine contract is made explicit: adapter-built rows are validated at the build boundary, externally supplied `samples=` remain pre-validated by the engine, and stamped rows are validated by the adapter immediately before replay mutation.

## 7. Deliberately not changed

This change does not alter:

- games/iteration (`64`)
- workers/contexts (`16 × 4`, total 64)
- batch cap/wait (`64 / 1 ms`)
- MCTS/search parameters
- komi (`0.5`)
- replay window or cap
- Adam settings or training budget
- self-play record serialization
- broker scheduling
- occupancy/drain scheduling

No new performance result is written to the Golden Standard until a controlled post-fix reproduction measures the actual wall-clock delta.
