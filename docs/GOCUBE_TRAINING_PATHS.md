# GoCube training path boundary

## Current GoCube V3

GoCube V3 is trained and searched only through the pinned KataGo contract:

```bash
python -m alphazero.envs.gocube.katago_train
```

Production wrappers use:

```bash
python -m alphazero.envs.gocube.hardened_train
```

The current contract uses Japanese V3 semantics, player-relative value
targets (`[WIN(side-to-move), LOSS(side-to-move), NO_RESULT]`), the pinned
KataGo search adapter, and `komi=0.5`. The adapter converts neural values to
absolute Black/White search values exactly once. `GoCube V3 + legacy MCTS`
is rejected at the MCTS boundary before self-play inference.

The old command is retired and fails closed:

```bash
python -m alphazero.envs.gocube.train
```

It must not be used for new self-play or training.

## Historical compatibility

`_LegacyGoGame`, `legacy_game_class`, V1/V2 adjudicators, and historical
checkpoint/model-contract loaders remain available for reading, evaluation,
or migration of old artifacts. They are not an authorization to create new
GoCube V3 training data. The V1/V2 classes retain their historical legacy
MCTS compatibility surface where existing evaluation workflows require it.

## Generic legacy AlphaZero

Framework-wide `MCTS.process_results()` and legacy behavior for non-GoCube
environments remain unchanged. The invariant is scoped to game classes marked
with the GoCube V3 contract marker, so Connect4 and other environments keep
their existing search behavior.

## Shared infrastructure

`alphazero.envs.gocube.training_common` owns the reusable Coach, CLI/config,
tensor validation, replay accounting, and telemetry components. The pinned
entrypoints import those components directly; neither production path imports
the retired executable module.

CI runs pinned Cube4 and Torus9 training-accounting smokes and verifies
non-empty self-play replay, row accounting, optimizer progress, provenance,
checkpoint contracts, and `komi=0.5`.

The deterministic value-contract reproduction is in
`tests/test_gocube_legacy_training_retirement.py`. The pre-existing Cube3
learning-sanity Arena artifacts are historical diagnostic evidence, not
post-fix acceptance evidence: their checkpoint manifest records
`searchContractId=gocube-search-contract-legacy`, which identifies the
retired path and reproduces the semantic mismatch. A fresh pinned sanity run
is recorded in `docs/GOCUBE_LEARNING_SANITY_20260909.md`; small runs are
diagnostic only and are not production-strength model quality evidence.
