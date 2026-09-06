# GoCube training-loop contract v1

This document covers the training-loop hardening layered on top of the pinned KataGo search/self-play contract from PR #32.

Pinned KataGo reference: `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

## Sample-clock learning rate

Pinned GoCube training uses checkpoint contract `gocube-sample-clock-v1`.

The optimizer clock is `total_training_samples`: after every successful optimizer step it increases by the actual batch row count (`boards.size(0)`), including a partial final batch. `total_optimizer_updates` is tracked separately. Neither counter depends on the outer self-play/training iteration number.

The effective LR is a function only of this persistent sample count:

1. linear warmup from `base_lr * gocube_lr_warmup_start_factor` to `base_lr` over `gocube_lr_warmup_samples`;
2. after warmup, each threshold in `gocube_lr_milestone_samples` multiplies LR by `gocube_lr_decay_gamma`.

The initial defaults are deliberately provisional: 2,000,000 warmup samples, start factor 0.05, sample milestones 20,000,000 and 40,000,000, gamma 0.1. They are configuration, not claims that these are optimal GoCube thresholds.

Checkpoint state contains the optimizer state, complete sample-scheduler state, explicit `total_training_samples`, `total_optimizer_updates`, current LR, and training-contract ID. A checkpoint without `gocube-sample-clock-v1` is rejected on resume instead of inferring a sample clock from its iteration number. Warmup therefore cannot silently restart after resume.

Telemetry records total training samples, optimizer updates, effective LR, samples since the last LR change, average/max pre-clip gradient norm, clipping event count, and clipping frequency. The same fields are added to the recorded iteration manifest after training.

## Gradient clipping

The optimizer and loss remain unchanged. After `loss.backward()` and before `optimizer.step()`, GoCube applies global gradient-norm clipping with `torch.nn.utils.clip_grad_norm_`. The cap is configured by `gocube_gradient_clip_norm` (default 5.0).

The default cap is a GoCube configuration value rather than a copied KataGo magnitude because KataGo's current caps depend strongly on its model/normalization/optimizer scale. The mechanism, placement, persistence and telemetry are the part standardized by this change.

## KataGo-style start diversification

Values copied from pinned `cpp/configs/training/selfplay8b20.cfg`:

- `earlyForkGameProb = 0.04`;
- `earlyForkGameExpectedMoveProp = 0.025`;
- `forkGameProb = 0.01`, sampled only if the early fork did not fire;
- `forkGameMinChoices = 3`;
- `earlyForkGameMaxChoices = 12`;
- `forkGameMaxChoices = 36`;
- `initGamesWithPolicy = true`;
- `policyInitAreaProp = 0.04`;
- `policyInitGammaShape = 1.0`;
- policy-init temperature defaults to KataGo's `1.0`.

Early fork depth is exponentially distributed with mean `0.025 * logicalPointCount`. Ordinary fork depth is uniform over the available played trajectory. `logicalPointCount` is the only Cube/Torus adaptation; no planar width/height assumption is introduced.

A fork stores the actual V3 state at the selected trajectory point. That state already contains pass, ko recap, repetition/cycle and phase history, so these histories are preserved rather than reconstructed from a flat board. The first experimental fork move is setup: it is never appended as a normal MCTS target. Standard policy-init moves are also setup and are excluded from training samples and ordinary move records. Training history starts only after setup completes.

### Deliberate differences from pinned KataGo

1. **Rules are fixed.** No ko/scoring/tax/suicide/button randomization is added. GoCube trains Japanese V3 only.
2. **Komi is fixed at 0.5.** No dynamic komi, compensation-after-policy-init, or fork komi compensation is used because komi/rules are not represented as NN inputs.
3. **Fork candidate ranking uses the existing root NN policy.** KataGo samples a small legal candidate set and chooses the candidate with best one-ply NN score for the side to move. GoCube preserves the pinned candidate-count distributions but chooses the highest raw-policy candidate in that random legal subset. This avoids adding a second per-candidate inference protocol to the existing coalesced worker service. The difference is explicit and test-covered.
4. **Plain fork pool is bounded at 1000 entries.** KataGo's seki pool is also bounded at 1000; GoCube uses the same bound for plain forks to prevent unbounded worker memory growth during long runs.
5. **Policy/setup inference reuses the existing self-play search cycle.** Only the raw root policy is consumed for the setup move; the setup position and move are not training targets.
6. **Smooth LR warmup is linear.** The pinned KataGo trainer historically uses sample-counted staged warmup factors. This PR uses the task-required smooth warmup while retaining the same sample-based clock principle.

Cleanup/encore training and seki forks from PR #32 remain separate curricula. Cleanup is sampled independently and, when selected after a policy/fork setup, begins only after that setup completes. Cleanup rebasing continues to clear history exactly as documented for the synthetic encore-training contract.

## Unchanged experiment dimensions

This change does not alter network architecture, optimizer type, loss weights, or search semantics from PR #32. In particular, the canonical run remains **50 regular MCTS sims and 20 fast sims**.
