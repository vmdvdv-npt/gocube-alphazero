# GoCube pinned KataGo exploration / policy-target port

## Source of truth

This block is derived from KataGo commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`, the same pinned commit used by the GoCube search contract introduced in PR #32.

Relevant KataGo behavior is taken from:

- `cpp/configs/training/selfplay8b20.cfg`
- `cpp/search/searchhelpers.cpp` (`computeDirichletAlphaDistribution`, root policy temperature/noise)
- `cpp/search/searchexplorehelpers.cpp` (`rootDesiredPerChildVisitsCoeff`, inverse PUCT reduction)
- `cpp/search/searchresults.cpp` (retrospective root play-selection-weight reduction)

Pinned values used by GoCube production self-play:

- `rootDirichletNoiseTotalConcentration = 10.83`
- `rootDirichletNoiseWeight = 0.25`
- `rootPolicyTemperatureEarly = 1.25`
- `rootPolicyTemperature = 1.10`
- temperature halflife = `19`
- `rootDesiredPerChildVisitsCoeff = 2.0`

## Ported semantics

At a self-play root GoCube now:

1. masks/normalizes the raw NN policy over legal actions;
2. applies KataGo's early root policy temperature;
3. applies KataGo's shaped Dirichlet distribution (half uniform alpha mass, half shaped by the low-policy log distribution);
4. uses `rootDesiredPerChildVisitsCoeff` during root selection to force underexplored policy children to receive search attention;
5. retains the actual raw visit counts for diagnostics;
6. derives the policy training target from retrospectively reduced root counts using KataGo's inverse-PUCT calculation, so exploration-forced overspend is not supervised as ordinary MCTS evidence.

The correction is intentionally separate from PASS/endgame behavior. Existing score utility, root ending bonus, FPU, PUCT, and PASS semantics from the pinned search contract are not redefined here.

Arena disables root temperature and Dirichlet noise but otherwise keeps the same deterministic search contract for both checkpoint opponents.

## Topology-specific substitution

KataGo's `interpolateEarly` scales elapsed halflives by `19 / sqrt(board_x * board_y)`. Cube and torus positions do not have a single planar `x*y` rectangle. GoCube substitutes the logical graph point count and therefore uses `19 / sqrt(logical_point_count)`.

This is the only topology-specific change in this exploration/policy-target block. Dirichlet shaping, mixture weight, desired-child-visits forcing, and retrospective inverse-PUCT reduction are topology-independent and retain the pinned formulas.

## Telemetry

For each regular self-play training position, game records can include:

- legal normalized NN root policy before exploration;
- policy after root temperature/noise;
- raw root visit counts;
- final policy training target;
- per-action and total retrospectively removed exploration visits;
- PASS-suppressed visits separately, so PASS/endgame suppression is not misreported as exploration correction.

Iteration manifests and TensorBoard contain aggregate raw visits, removed exploration visits, target visits, and removed-visit fraction.
