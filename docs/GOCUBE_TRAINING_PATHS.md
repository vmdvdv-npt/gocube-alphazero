# GoCube training path boundary

## Komi policy

The project-wide komi policy is defined in `docs/KOMI_POLICY.md`. `0.5` is the
current default/canonical baseline, not a permanently proven fair value for
Torus. Explicit finite alternatives are permitted in new/reference/research
components when recorded as part of the rules/evaluation identity. Legacy
`7.5` is forbidden in current runtime paths and must fail closed with owner
escalation rather than being silently coerced.

Historical/frozen experiment paths may still pin `0.5` locally for exact
reproducibility. Those local assertions do not define the policy for the new
Torus path.

## Current GoCube V3

GoCube V3 is trained and searched only through the pinned KataGo contract:

```bash
python -m alphazero.envs.gocube.katago_train
```

Production wrappers use:

```bash
python -m alphazero.envs.gocube.hardened_train
```

This existing Japanese-V3 stack was defined with a historical `komi=0.5`
baseline, player-relative value targets
(`[WIN(side-to-move), LOSS(side-to-move), NO_RESULT]`), and the pinned KataGo
search adapter. Its path-specific 0.5 checks are retained only for
reproducibility and must not be inherited by the new Torus reference/training
architecture. The adapter converts neural values to absolute Black/White
search values exactly once. `GoCube V3 + legacy MCTS` is rejected at the MCTS
boundary before self-play inference.

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

An old artifact that exposes komi `7.5` must not be automatically upgraded or
registered as a current model. Stop and contact the project owner so its true
historical semantics can be handled explicitly.

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

Current CI retains historical V3/Cube-4 assertions where they are part of a
frozen experiment contract. The project-wide policy tests separately verify
that the default is `0.5`, explicit finite nonlegacy values are accepted by the
shared validator, and legacy `7.5` fails closed.

The deterministic value-contract reproduction is in
`tests/test_gocube_legacy_training_retirement.py`. The pre-existing Cube3
learning-sanity Arena artifacts are historical diagnostic evidence, not
post-fix acceptance evidence: their checkpoint manifest records
`searchContractId=gocube-search-contract-legacy`, which identifies the
retired path and reproduces the semantic mismatch. A fresh pinned sanity run
is recorded in `docs/GOCUBE_LEARNING_SANITY_20260909.md`; small runs are
diagnostic only and are not production-strength model quality evidence.
