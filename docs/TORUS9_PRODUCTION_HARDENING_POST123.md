# Torus9 production hardening after PR #123

Date: 2026-09-17  
Branch: `codex/torus9-production-hardening-post123`  
Implementation commit at the successful smoke: `8f17847beadc083fb73f8a2102229b707d6dd2d9`

## Result

The Torus9 production path is hardened and the successful Legion smoke is
archived by move at:

`runs/torus9/archive/torus9-post123-smoke-20260917-validated-final/`

The archived lineage ended `COMPLETED` after two committed generations and a
same-lineage Arena. Its manifest is `ARCHIVED`; no checkpoint was copied to a
different lineage.

## Implemented changes

- Added one strict Torus9 profile loader/validator. It checks the scientific
  sections and the actual profile/content fingerprint before self-play,
  training, or Arena; stale embedded fingerprints cannot mask payload drift.
- Added semantic progress heartbeat fields with rate-limited writes. Liveness
  and progress are independent; self-play reports completed games, replay
  reports target-build/validation rows, training reports optimizer steps, and
  Arena reports completed games.
- Added phase timing for restore, replay load/validation/update, target build,
  serialization, checkpoint write/reload, and publication. Committed replay
  identities carry file SHA, row count, canonical fingerprint, and validation
  schema. Continuation reloads use catalog evidence and do not repeat the
  canonical fingerprint pass.
- Added an append-only committed artifact catalog. Arena verifies only the
  candidate, reference, transaction/state, and bounded metadata needed for the
  current comparison; it does not walk or rehash the historical tree.
- Added NVIDIA telemetry resolution in the order environment override, PATH,
  then `/usr/lib/wsl/lib/nvidia-smi`, with explicit unavailable, command, and
  parse statuses and per-GPU utilization/VRAM/temperature/power fields.
- Added regression coverage for profile drift, heartbeat progress and stalls,
  catalog mutation/complexity, and telemetry resolver/error cases.

## Follow-up commit recovery

The generation transaction is the durable commit journal.  On explicit
recovery, the orchestrator now completes an interrupted tail of the commit
idempotently:

- a `COMMITTED` transaction without a catalog generation re-validates and
  publishes only that generation's result identities;
- an existing catalog generation is checked against the transaction identity,
  artifact set, and bounded file identities;
- runtime state and manifest catalog/checkpoint records are then advanced
  together, leaving the lineage in `RECOVERY_REQUIRED` until the operator
  explicitly runs `resume`.

This covers both process-stop windows: after the transaction write and after
catalog/state writes but before the manifest write.  Regression tests inject
both stops and verify that the same lineage resumes at the next generation;
an unexplained catalog mutation remains fail-closed.  The child process-group
teardown and self-play worker cleanup from PR #124, along with the stabilized
soft-stop timing test from PR #125, are included in the combined branch.

## Verification

```text
329 passed, 1 skipped
PYTHONPATH=. .venv/bin/pytest -q
```

Exact M16→M17 parity (`tools/torus9_training_engine_parity.py`, CUDA):

- model parity: PASS; canonical/reproduced M17 SHA
  `sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5`
- max parameter difference: `0`
- optimizer parity: PASS; max absolute delta `0`
- fresh replay order/SHA: equal
- rolling replay order/fingerprint: equal
- metadata: exact
- 80 optimizer steps and 5120 samples; M18 was not created and canonical M17
  was not mutated

## Legion smoke

Run-spec fingerprint:
`sha256:90bbf7dd3e4ca458989dca79e53658acebb0caeeae59da6687728e6a2b3595b2`

Profile fingerprint:
`sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`

The immutable workload used Torus9 64 self-play games, 16 workers × 4 active
games, 64 contexts, cap64/wait1ms, CUDA, rolling replay 3 generations/cap
20,000, Adam batch64, and 80 optimizer steps per generation. The Golden
scientific parameters were unchanged.

Lifecycle evidence:

1. Detached `start` returned only after the startup handshake.
2. Live `status`/`status --watch` showed incremental self-play values such as
   `5/64`, `25/64`, `57/64`, and `64/64 games`; replay reached `6389/6389
   rows`; Arena reached `64/64 games`.
3. `stop` during M1 produced `SOFT_STOPPED` after the current safe unit;
   generation 1 committed and no Arena started.
4. `resume` reused the same directory and run-spec, restored M1 model,
   optimizer, replay, generation clock, RNG, and progress bookkeeping.
5. M2 committed and periodic Arena ran at the same lineage location. Final
   state was `COMPLETED`, with no automatic restarts and no technical games.

Generation evidence:

| Generation | Self-play | Replay positions | Optimizer | Samples | Mean loss | Replay mode |
|---:|---:|---:|---:|---:|---:|---|
| M1 | 64 games / 4647 moves | 4647 | 80 | 5120 | 4.96708 | initial-empty |
| M2 | 64 games / 6389 moves | 11036 | 80 | 10240 cumulative | 4.81008 | catalog-evidence |

Observed M2 timing includes restore `14.61 s`, replay file load `1.92 s`,
replay validation bookkeeping `11.69 s`, replay update `37.56 s`, replay
update+validation `102.34 s`, target build `32.97 s`, training `7.39 s`, and
serialization `4.45 s`. These are wall-clock smoke measurements, not a
controlled benchmark; the important integrity result is that immutable replay
semantic validation was not repeated on continuation reload.

Arena M2 result:

- W/L/D: `59/5/0`; 64 games; 0 technical games
- status: `PASS`; 9586 moves; `15.25 moves/s`; wall time `628.47 s`
- execution: CUDA, 16 workers × 4 games/worker, cap64/wait1ms
- candidate/reference stayed unchanged before and after Arena; catalog had 20
  committed artifact entries across generations 1 and 2, and both generation
  transactions were `COMMITTED`

GPU telemetry was collected inside the Arena workload:

- resolver: `wsl_fallback`
- binary: `/usr/lib/wsl/lib/nvidia-smi`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- 632 samples; status `ok` for all samples
- utilization mean/max: `33.53%` / `60%`
- VRAM used mean: `779.7 MiB` of `6144 MiB`
- temperature mean: `56.17 °C`; power mean: `19.59 W`

The first smoke attempt with an additional `min_mean_inference_batch_rows=16`
gate completed 64/64 Arena games with zero technical games but correctly failed
closed on observed mean batch size `7.19`. Its heavy data was removed only
after a required discarded-run record was written. The subsequent code-pin
attempts were also recorded and discarded; they provide evidence that dirty
trees and commit drift are rejected before production work starts.
