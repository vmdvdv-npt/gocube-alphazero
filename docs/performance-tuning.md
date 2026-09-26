# Performance tuning extraction

The standalone mode is `mode: performance_tuning` in the existing Orchestrator
V2 run-spec. It uses the normal `Torus9ProductionLineage` owner and the existing
`ProductionTrainOne`/`SupervisorV2` boundary. The tuner owns the measurement
plan and evidence; it does not create a process supervisor.

Only these execution values are allowed in v1:

| Plan field | Production mapping | Validation |
| --- | --- | --- |
| `active_games_per_worker` | `execution_overrides.active_games_per_worker` | positive integer |
| `total_active_contexts` | `execution_overrides.total_active_contexts` | positive integer |
| `workers` | fixed plan constraint; never an override | positive integer and capacity check |

Scientific parameters are never copied from a profile into the effective config.
The scientific-contract fingerprint excludes only the execution section and must
match before and after resolving a profile. Learning rate, MCTS simulations,
replay, games per generation, optimizer steps, batch size, arena rules, seed,
and topology therefore remain immutable for a tuning plan.

## Telemetry contract

The source is the committed generation summary. A committed marker and checkpoint
identity are required before an observation is accepted; heartbeat/progress alone
is not evidence.

| Field | Source and unit | Missing/invalid | Combination and use |
| --- | --- | --- | --- |
| `games` | `orchestrator_selfplay.games`, games | unstable | must equal expected generation count |
| `games_per_hour` | `orchestrator_selfplay.games_per_hour`, games/hour | unstable; NaN/∞ rejected | arithmetic mean among stable observations; primary ranking |
| `selfplay_wall_time_sec` | `orchestrator_selfplay.selfplay_time_sec`, seconds | unstable; NaN/∞/non-positive rejected | arithmetic mean; tie-break after games/hour |
| `execution` | `orchestrator_selfplay.execution`, integer topology values | unstable | must equal planned workers/contexts |
| `cycle_wall_time_sec` | adjacent persisted train-one request timestamps, seconds | allowed diagnostic | not used as self-play speed; phases differ |
| `moves`, `moves_per_sec` | self-play summary, moves / moves per second | allowed diagnostic unless needed for a derived field | retained for provenance |
| GPU/CPU utilization and power | first finite value from summary, inference, timing | allowed diagnostic | never used by the v1 selector |
| technical/invalid games and stalls | summary/timing/inference counters | non-zero makes observation unstable | no averaging of unstable observations |

The selection rule is intentionally unchanged from the inline sweep:
mean `games_per_hour`, then smaller mean self-play wall time, with the prior
insertion-order tie break. A failed label is excluded. If no stable evidence is
available, the selected profile is explicitly marked
`explicit_baseline_fallback`; it is not described as measured-best.

Standalone measurements are sequential generations on an evolving checkpoint
lineage. The report records parent and measurement checkpoint refs and states
this limitation; it is not a same-checkpoint A/B benchmark. Different execution
schedules are not promised to produce bit-identical self-play order or weights.

## Run-spec shape

```json
{
  "schema": "gocube-orchestrator-v2-run-spec-v1",
  "mode": "performance_tuning",
  "performance_tuning": {
    "tuning_id": "torus9-concurrency-20260926",
    "lineage_id": "torus9-tuning-20260926",
    "topology": "torus9",
    "parent_checkpoint": {"...": "existing immutable CheckpointRef"},
    "effective_config": {"...": "existing EffectiveConfig"},
    "workers": 16,
    "baseline": {"label": "baseline", "active_games_per_worker": 4, "total_active_contexts": 64},
    "modes": [{"label": "six-by-96", "active_games_per_worker": 6, "total_active_contexts": 96}],
    "measurement_budget": {"max_actions": 3, "baseline_repetitions": 1, "repetitions_per_mode": 1},
    "finish_behavior": "export_profile"
  }
}
```

`--dry-run` is available as the `dry_run: true` plan flag and validates the
same plan without creating a child process, checkpoint, tuning state, or event.
The first release supports `export_profile`; standalone `continue_training` is
rejected rather than implying unbounded additional training.

## Legacy compatibility and recovery

The inline `self_play_concurrency_sweep` is parsed into the same `Mode` and
policy implementation through `LegacyConcurrencyAdapter`. Its canonical state
remains the existing `performance_sweep` object and its report remains
`metrics/self-play-concurrency-sweep-v1.json`. Unknown fields are preserved.
The historical `continue_after_sweep` field is serialized and accepted, but the
researched version did not read it as a stop condition in the main loop; this
extraction preserves that observed behavior. Changing that semantic is a
separate compatibility change.

| Legacy evidence | New adapter action | New computation |
| --- | --- | --- |
| no sweep started | baseline sequence remains pending | no standalone tuning files |
| baseline evidence | read committed summary and preserve ref | `assess_observation` |
| middle of modes | preserve `next_mode_index` | `choose_next` equivalent |
| failed mode | preserve label/generation/reason | structured category at adapter edge |
| pending baseline retry | reuse same generation and owner | no generation increment before retry |
| selected mode | preserve selected mode and state | deterministic policy re-read |
| completed run | reuse committed state/report | no new computation |
| soft stop | leave durable stop in place | existing continuous safe boundary |

Resume windows A-H are fail-closed: a saved action/profile is reused, a live
child is reattached by the existing supervisor, a committed generation is
recovered before observation publication, selection is recalculated without a
new generation, and a report is regenerated from canonical observations. A
state/request/commit identity disagreement stops with an inconsistency error.
Explicit stop, configuration, integrity, and unknown failures do not trigger
automatic baseline recovery. A technical retry has a distinct durable intent
and consumes explicit measurement budget.

Rollback keeps the old state/report and checkpoint refs; it does not rewind to an
older checkpoint. Before rollout, copy old fixtures and record the code commit,
canonical state, selected profile, and refs. Roll out in this order: ordinary
fixed-profile run, controlled standalone tuning, then a resumed inline sweep.
