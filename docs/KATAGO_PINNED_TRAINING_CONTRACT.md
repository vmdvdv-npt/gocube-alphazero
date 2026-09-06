# GoCube pinned training-loop contract

This document covers the training-loop safeguards and self-play diversification layered on top of the pinned KataGo search port from PR #32.

Reference KataGo commit: `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

## Sample-based training clock

Pinned GoCube training now uses `gocube-sample-clock-v1`. The persistent clock is `total_training_samples`, incremented by the **actual batch size only after a successful `optimizer.step()`**. A partial final batch therefore advances the clock by its real row count rather than by the configured batch size.

Learning rate is a pure function of this sample clock. External AlphaZero iteration numbers no longer advance the LR scheduler.

Checkpoint state contains:

- `total_training_samples`;
- `total_optimizer_updates`;
- optimizer state;
- sample-clock scheduler state, including its current sample position and last LR-change sample.

A checkpoint that predates `gocube-sample-clock-v1` is rejected on training resume. GoCube deliberately does **not** infer an approximate sample count from an old iteration number.

### LR warmup and decay

Warmup is linear in samples from `lr * warmup_start_factor` to the configured base `lr`. Defaults in the pinned GoCube entrypoint are:

- warmup samples: `100000`;
- warmup start factor: `0.05`;
- post-warmup sample milestones: none by default;
- milestone gamma: `0.1`.

The empty default milestone list is intentional. This PR changes the clock semantics but does not claim that a particular GoCube decay threshold has already been tuned. Long runs can provide explicit comma-separated sample milestones through `--lr-sample-milestones` without changing the training contract.

Pinned KataGo also maintains a sample-based `global_step_samples`, but its historical warmup is stepped. GoCube deliberately uses a **smooth linear warmup** because this change set explicitly requires smooth warmup.

## Gradient clipping

Global parameter gradient norm is measured after `backward()` and before `optimizer.step()`, then clipped with `torch.nn.utils.clip_grad_norm_`.

Default max norm: `5.0`, configurable with `--gradient-clip-norm`.

The optimizer type, optimizer hyperparameters, network architecture, and loss weights are unchanged.

Telemetry records the pre-clipping norm, clipping event count, clipping check count, and clipping frequency.

## Training telemetry

Each pinned iteration writes `training-metrics.json` next to that iteration's game records. It contains, among the existing training budget metrics:

- total training samples;
- total optimizer updates;
- effective LR;
- samples since the last effective LR change;
- gradient norm before clipping;
- clipping event/check counts and frequency.

The same numeric values are emitted to TensorBoard and the iteration console summary.

## KataGo-style self-play diversification

The pinned `selfplay8b20.cfg` settings are used:

- `earlyForkGameProb = 0.04`;
- `earlyForkGameExpectedMoveProp = 0.025`;
- `forkGameProb = 0.01`;
- `forkGameMinChoices = 3`;
- `earlyForkGameMaxChoices = 12`;
- `forkGameMaxChoices = 36`;
- `initGamesWithPolicy = true`;
- `policyInitAreaProp = 0.04`;
- `policyInitGammaShape = 1.0`;
- `policyInitAreaTemperature = 1.0` (the pinned default);
- existing `sekiForkHackProb = 0.02` remains unchanged.

Early-fork depth is sampled from KataGo's exponential rule with mean `0.025 * logicalPointCount`. Ordinary fork depth is uniform over the finished game's move history. The ordinary probability is conditional on the early fork not firing, so its unconditional expected rate is `0.96 * 0.01 = 0.0096` per eligible source game.

Cube/Torus adaptation uses only `logicalPointCount` and the existing graph topology/rules implementation. No planar-board coordinates are introduced.

### History and target handling

Fork source positions retain the real V3 state at the fork point, including ko/pass/cycle history. The experimental fork move is treated as setup and is not appended to the MCTS training-target accumulator. Training begins only after the experimental move.

Policy-initialization moves likewise use legal V3 transitions and retain their resulting history, but the prelude moves are not emitted as ordinary MCTS targets.

Cleanup training remains different by design: PR #32 already mirrors KataGo's cleanup rebase by preserving the initialized board/player/capture counts while clearing the history-dependent fields required by the synthetic cleanup phase.

Seki forks remain a separate pool and retain the PR #32 semantics.

Telemetry records normal starts, early forks, ordinary forks, policy-initialized starts, and average fork depth.

## Deliberate differences from pinned KataGo

1. **Fixed rules and komi.** GoCube keeps Japanese V3 and komi `0.5`. Rules randomization, `komiAuto`, fork komi compensation, and dynamic komi are not used because rules/komi are not represented as NN inputs.
2. **Fork experimental-move ranking.** KataGo samples a small legal candidate set and evaluates one-ply candidate positions with its value/score evaluator, choosing the best score for the player. GoCube samples the same configured candidate-count range but uses the cached raw policy over that random legal subset. This avoids creating a second inference protocol inside the batched worker loop. The experimental move remains setup-only. This is the principal deliberate self-play difference in this change set.
3. **No recursive plain-fork seeding.** A game that itself began from a plain or seki fork does not seed another plain fork. This bounds feedback between fork curricula. Normal completed games remain the source distribution.
4. **Smooth warmup.** KataGo's pinned historical trainer uses sample-counted stepped warmup; GoCube uses sample-counted linear warmup as required by this change set.
5. **No claimed universal decay thresholds.** Sample-milestone support is implemented, but the pinned GoCube entrypoint leaves post-warmup milestones empty until GoCube-specific scale is measured.

## Search-budget invariant

This change set does not alter the requested search budgets:

- regular self-play: `50` simulations;
- fast self-play: `20` simulations.

Search utility, cleanup training, seki forks, pass/endgame semantics, and the rest of PR #32 remain on their pinned contracts.
