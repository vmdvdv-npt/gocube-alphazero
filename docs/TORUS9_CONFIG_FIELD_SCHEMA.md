# Torus 9x9 canonical config field names

`configs/gocube/torus9_golden_current_v3.json` is the canonical naming source for new Torus 9x9 work. Historical profiles are preserved byte-for-byte for reproducibility, so they may still contain old names.

The machine-readable registry is `configs/gocube/torus9_field_schema_v1.json`. Every old name in that registry is explicitly marked `legacy-alias` or `legacy-retired`.

| Legacy field | Canonical field | Status |
| --- | --- | --- |
| `self_play.simulations` | `self_play.mcts_simulations` | legacy-alias |
| `self_play.batch_size` | `self_play.existing_batch_size` | legacy-alias |
| `arena.simulations` | `arena.mcts_simulations` | legacy-alias |
| `training.scheduler` | `training.lr_scheduler` | legacy-alias |
| `training.gating` | `training.model_gating` | legacy-alias |
| `replay.maximum_positions` | `replay.cap` | legacy-alias |
| `replay.policy` | `replay_policy` | legacy-alias |
| `training.replay` | `replay_policy` | legacy-alias |
| `self_play.fast_sims` | none | legacy-retired |
| `arena.one_game_per_process` | none | legacy-retired |
| `arena.technical_fail_closed` | none | legacy-retired |
| `rules.legacy_komi_7_5_rejected` | none | legacy-retired |

## Rules

1. New/current profiles use canonical names only.
2. Historical files are not rewritten just to rename fields; their fingerprints remain valid evidence.
3. `migrate_legacy_field_names()` may be used when reading historical data for comparison or tooling. It changes field names only; it does not claim scientific equivalence between historical and current profiles.
4. If both a legacy alias and its canonical name are present with different values, migration fails closed with `LegacyFieldConflictError`.
5. Retired fields are accepted only as historical migration input and are removed by migration. They must not appear in current profiles.
6. Komi remains exactly `0.5`. The historical `7.5` sentinel is not a valid komi value and remains fail-closed.

CI tests verify that the current profile contains no registered legacy field names, that the historical profile still validates unchanged, that the registry and Python migration table agree, and that conflicting aliases cannot silently override canonical values.
