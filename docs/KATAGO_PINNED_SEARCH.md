# Pinned KataGo search port for GoCube

## Scope

This branch starts from `main` and deliberately does **not** inherit the abandoned `fix/gocube-score-aware-pass-v1` search/recovery stack.

The rules and terminal state machine remain the existing GoCube Japanese V3 implementation, pinned to KataGo commit:

`f6bc4b19a1686caa2d088b56251e8c11c8be6d51`

The new search layer ports the relevant behavior from that same pinned source. Cube and Torus topology differences remain isolated behind `Topology.neighbor_indices(...)` and the existing graph Benson/pass-alive implementation.

## Pinned self-play search semantics

The production GoCube search mode is `katago-pinned-f6bc4b19`.

From `cpp/configs/training/selfplay8b20.cfg` and the pinned search sources:

- `winLossUtilityFactor = 1.0`
- `staticScoreUtilityFactor = 0.0`
- `dynamicScoreUtilityFactor = 0.30`
- `dynamicScoreCenterZeroWeight = 0.25`
- `dynamicScoreCenterScale = 0.50`
- `cpuctExploration = 1.10`
- `cpuctExplorationLog = 0.0`
- `fpuReductionMax = 0.20`
- `rootFpuReductionMax = 0.0`
- `fpuParentWeightByVisitedPolicy = true`
- `fpuParentWeightByVisitedPolicyPow = 2.0`
- `rootEndingBonusPoints = 0.50`

`selfplay8b20.cfg` does **not** set `conservativePass` or `fillDameBeforePass`. At the pinned commit, KataGo's `SearchParams` defaults both to `false`, so the GoCube pinned self-play profile now also uses:

- `conservativePass = false`
- `fillDameBeforePass = false`

KataGo enables these behaviors in some GTP/analysis configurations; those analysis defaults must not be mistaken for self-play training settings.

Search utility is stored from White's perspective, like KataGo. Selection converts it to the player-to-move perspective only at the final PUCT comparison.

The PUCT exploration term follows KataGo's `sqrt(totalChildWeight + 0.01)` form. FPU uses visited policy mass and the pinned parent-NN/parent-average blend rather than the previous framework-only FPU formula.

## Score utility

GoCube's score head predicts one normalized Black-minus-White scalar. Search converts it to White-minus-Black score points before combining it with result utility.

The dynamic score center follows the pinned KataGo formula exactly in the important detail that the cap is relative to the current `expectedScore`, not relative to zero:

1. `center = expectedScore * (1 - zeroWeight)`
2. cap the center to `expectedScore +/- sqrt(pointCount) * dynamicScoreCenterScale`

KataGo models score variance as well as score mean. GoCube currently has only a scalar score head, so this port uses the zero-score-stdev specialization of KataGo's smooth score-value function. It does not invent a variance estimate.

## PASS and root ending behavior

Ownership is **not** a generic veto or permission signal.

It is consumed by the implemented root-ending bonus/penalty logic. The root ending calculation uses the pinned `0.95` ownership extreme and `0.05` tail, the territory PASS penalty of `2/3 * rootEndingBonusPoints`, opponent adjacency, captures, and a graph-topology adaptation of `isNonPassAliveSelfConnection` based on the existing V3 pass-alive analysis.

Root ending bonuses are precomputed once per root, then converted through score utility during PUCT selection.

The code also retains an implementation of KataGo's `fillDameBeforePass` heuristic and a GoCube observation adaptation for `conservativePass` for explicit non-self-play use. Neither is enabled by the pinned self-play training profile, matching KataGo's effective `selfplay8b20.cfg` settings.

`rootPruneUselessMoves` is **not yet enabled**. KataGo's implementation requires the last four opponent moves explicitly; GoCube V3 does not currently retain the equivalent ordered move history in semantic state. Rather than invent a replacement threshold, the behavior is left disabled until the required state is represented directly.

## Cleanup PASS-for-ko

The V3 rules engine implements both pass-for-ko forms in pinned KataGo. In cleanup, a player can lift a ko-recapture block either by choosing the blocked opposing single stone in atari, or by choosing the empty ko-capture point whose unique capturable one-stone target is that blocked stone. Both consume the turn, clear the relevant block, and leave the board unchanged. The normal `point_count + PASS` action space is sufficient; no extra action is added.

## Terminal treatment

No new terminal heuristic is introduced. Search uses the existing pinned V3 rules state machine:

`MAIN -> CLEANUP_1 -> CLEANUP_2 -> SCORED`, plus `NO_RESULT`.

For a scored terminal, search uses the exact final score and exact result. For `NO_RESULT`, result utility is zero with the pinned `noResultUtilityForWhite = 0`, and no artificial score is supplied.

## Tree reuse

Dynamic score utility is centered on the current root. Retaining subtree Q values after a move would mix utilities computed with two different score centers unless every retained statistic were re-centered. This first clean port therefore resets score-aware MCTS after each played move. This is slower than full KataGo tree reuse but semantically safe; re-centered tree reuse can be added separately without changing the search contract.

## Batching

Policy, value, ownership, and score are produced by one network forward pass. Existing cross-worker coalesced inference is preserved: ready worker batches are concatenated, evaluated once, then all four heads are copied back to worker-owned shared tensors before releasing the workers.

## Cube 4 from-scratch pilot

The dedicated entrypoint refuses to reuse an existing run namespace unless explicitly overridden.

Canonical pilot:

```bash
tools/run_cube4_katago_from_scratch.sh
```

Defaults:

- topology: Cube
- face size: `4x4`, six faces = 96 logical points
- workers: `16`
- regular MCTS sims: `50`
- arena sims: `50`
- fast-search probability: `0.25`
- games per iteration: `256`
- iterations: `8`
- random-MCTS warmup: disabled; iteration 0 starts from a randomly initialized network and policy-guided self-play
- `conservativePass = false`
- `fillDameBeforePass = false`

Search noise, move temperature, games/iteration, network size, optimizer, learning rate, and other training-scale parameters remain experiment dimensions rather than being claimed universal KataGo constants.
