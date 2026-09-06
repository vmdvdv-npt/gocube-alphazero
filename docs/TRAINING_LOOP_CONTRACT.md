# GoCube production training-loop contract

This document covers the production-training hardening layered on top of the pinned KataGo search/self-play contract from PR #32 and the replay/exploration/Arena contract from PR #34.

Pinned KataGo reference: `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.

GoCube production komi is fixed at **0.5**.

## Sample-clock learning rate

Pinned GoCube training uses checkpoint contract `gocube-sample-clock-v2`.

The optimizer clock is `total_training_samples`: after every successful optimizer step it increases by the actual batch row count (`boards.size(0)`), including a partial final batch. `total_optimizer_updates` is tracked separately. Neither counter depends on the outer self-play/training iteration number.

The effective LR is a function only of this persistent sample count:

1. linear warmup from `base_lr * gocube_lr_warmup_start_factor` to `base_lr` over `gocube_lr_warmup_samples`;
2. after warmup, each threshold in `gocube_lr_milestone_samples` multiplies LR by `gocube_lr_decay_gamma`.

The initial defaults remain configuration rather than claims of optimality: 2,000,000 warmup samples, start factor 0.05, sample milestones 20,000,000 and 40,000,000, gamma 0.1.

Checkpoint state contains optimizer state, complete sample-scheduler state, explicit `total_training_samples`, `total_optimizer_updates`, current LR, and training-contract ID. A checkpoint without the current sample-clock contract is rejected rather than inferring a sample clock from the iteration number.

## Sample-based replay/training ratio

The optimizer budget is derived only from newly produced self-play samples:

`planned_training_samples = ceil(new_selfplay_samples * train_samples_per_new_sample)`.

The replay window is only a random sampling source. Growing replay history does not increase the optimizer budget. Production uses replacement sampling for exactly the planned number of training rows, including a partial final batch when needed.

Telemetry includes new samples, replay-window samples, planned/actual training samples, optimizer steps, effective train/new ratio, passes over the replay window, total training samples, LR and gradient statistics.

## Gradient clipping

After `loss.backward()` and before `optimizer.step()`, production training applies global gradient-norm clipping with `torch.nn.utils.clip_grad_norm_`. The configured default cap remains 5.0 and is checkpoint-contract metadata.

## Pinned KataGo root exploration and policy target

Production root exploration is versioned as `katago-pinned-exploration-v2` and keeps the pinned PR #34 mechanisms:

- shaped Dirichlet noise: total concentration 10.83, weight 0.25;
- root NN-policy temperature 1.25 -> 1.10 with pinned `interpolateEarly`;
- `rootDesiredPerChildVisitsCoeff = 2.0`;
- retrospective inverse-PUCT reduction before policy supervision.

For Cube/Torus, KataGo's `sqrt(x*y)` interpolation scale is replaced only by `sqrt(logical_point_count)`.

### Chosen-move temperature

The actually played self-play move now follows pinned KataGo:

- `chosenMoveTemperatureEarly = 0.75`;
- `chosenMoveTemperature = 0.15`;
- `chosenMoveTemperatureHalflife = 19`;
- the same logical-point-count interpolation used for root policy temperature.

This is separate from root NN-policy temperature. The old framework temperature scaler depended on `GameState.max_turns()` and therefore left GoCube self-play at temperature 1.0 for the whole game. Production no longer uses that behavior for pinned self-play.

### Value-weighted node aggregation

Pinned `valueWeightExponent = 0.5` is now part of the production search contract.

Each node retains utility mean, utility-square mean, total statistical weight and squared-weight sum. During parent recomputation, children whose self-utility is poor relative to the current child average are downweighted using KataGo's normal-CDF weighting and the adjusted weights are renormalized to preserve total child weight. PUCT uses the resulting KataGo-style child `weightSum`, not raw visit count, for search weight and exploration denominators.

The production search is a tree rather than KataGo graph search, so an edge's visit ratio is one and `getChildWeight(edgeVisits)` reduces to the child's aggregated `weightSum`. This is the explicit no-transposition specialization; no synthetic transposition statistics are introduced.

### LCB play selection

Pinned self-play parameters are:

- `useLcbForSelection = true`;
- `lcbStdevs = 5.0`;
- `minVisitPropForLCB = 0.15`;
- `chosenMoveSubtract = 0`;
- `chosenMovePrune = 1`.

LCB uses the stored utility variance and effective sample size (`weightSum^2 / weightSqSum`), including KataGo's variance prior and root ending-score utility adjustment.

KataGo deliberately disables LCB while choosing the **actual self-play move**, then restores it when extracting the **policy training target**. GoCube mirrors that split: the chosen-move distribution uses retrospective weights without LCB, while the policy target uses LCB-adjusted play-selection weights. Deterministic Arena uses LCB and move temperature 0.

## KataGo-style start diversification

Values copied from pinned `cpp/configs/training/selfplay8b20.cfg` remain:

- `earlyForkGameProb = 0.04`;
- `earlyForkGameExpectedMoveProp = 0.025`;
- `forkGameProb = 0.01` after early-fork rejection;
- candidate limits 3..12 early and 3..36 ordinary;
- `initGamesWithPolicy = true`;
- `policyInitAreaProp = 0.04`;
- policy-init gamma shape 1.0 and temperature 1.0.

Rules remain Japanese V3 only and komi remains fixed at 0.5. No unrepresented rules or dynamic komi are injected into NN training.

## Crash-safe checkpoint and replay recovery

The unattended production entrypoint uses recovery contract `gocube-atomic-recovery-v1`.

### Checkpoints

A checkpoint is first written to a hidden sibling staging file, flushed with `fsync`, and then promoted to `iteration-NNNN.pkl` with atomic `os.replace`; the containing directory is also synced. A crash therefore cannot expose a partially written new checkpoint under the final iteration filename.

Resume does not use `len(glob("iteration-*.pkl"))`. It scans the contiguous sequence `iteration-0000.pkl`, `iteration-0001.pkl`, ... and structurally loads each file. The first missing or unreadable checkpoint terminates the resumable prefix. Any later files are ignored, and the selected checkpoint still has to pass the normal rules/search/training contract validation when loaded.

### Replay tensors

The six replay tensors for an iteration are first produced in a staging directory on the same filesystem. Only after all six are complete are they atomically promoted to the run directory. A versioned `iteration-NNNN-complete.json` marker is written **last**.

Replay loading requires this completion marker and checks its row count against all six tensors. If a process dies after only some files have been promoted, the missing marker makes the iteration invalid and it is ignored/rebuilt rather than silently entering the replay window.

## Production entrypoint

The hardened production module is:

```bash
python -m alphazero.envs.gocube.production_hardening ...
```

The canonical Cube 4 launcher `tools/run_cube4_katago_from_scratch.sh` uses this module. It remains from-scratch by default; explicit resume requires `--allow-existing-run` and all checkpoint contracts must match.

## Unchanged experiment dimensions

This hardening does not change the selected production regular-search budget (**50 regular sims**), fast-search budget (**20 sims**), network architecture, optimizer type, loss weights, fixed komi 0.5, or PR #32 rules/PASS/FPU semantics. It also does not introduce KataGo graph search/transpositions, rules randomization, dynamic komi, SWA, or new NN heads.
