# GoCube pinned KataGo exploration / policy-target port

## Source of truth

This block is derived from KataGo commit `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`, the same pinned commit used by the GoCube search contract introduced in PR #32.

Production contract: `katago-pinned-exploration-v2`.

Relevant KataGo behavior comes from `selfplay8b20.cfg`, `searchhelpers.cpp`, `searchexplorehelpers.cpp`, `searchupdatehelpers.cpp`, `searchresults.cpp`, and `play.cpp`.

Pinned production values:

- root Dirichlet total concentration 10.83, weight 0.25;
- root policy temperature 1.25 -> 1.10, halflife 19;
- root desired child visits coeff 2.0;
- chosen move temperature 0.75 -> 0.15, halflife 19;
- chosen move subtract 0, prune 1;
- value weight exponent 0.5;
- LCB enabled, 5.0 stdevs, minimum visit proportion 0.15.

## Search and policy-target order

At a self-play root GoCube now:

1. masks/normalizes raw NN policy over legal actions;
2. applies pinned early root-policy temperature;
3. applies shaped Dirichlet root noise;
4. searches using root desired-child forcing, KataGo PUCT/FPU and value-weighted child statistics;
5. retains raw edge visits for diagnostics;
6. obtains root play-selection weights from child `weightSum`;
7. retrospectively reduces over-explored child weights with inverse PUCT;
8. applies LCB for policy-target extraction;
9. applies chosen-move prune/subtract;
10. normalizes the resulting weights as the training policy target.

For the actual self-play action, step 8 is deliberately skipped, matching KataGo's `play.cpp` self-play hack that temporarily disables LCB for move choice. The action is then sampled with the chosen-move temperature schedule 0.75 -> 0.15. Policy supervision still uses LCB.

## Value aggregation

The old GoCube backup gave every leaf equal statistical weight and averaged utility directly. Production now retains utility-square and effective-weight statistics and ports pinned `valueWeightExponent=0.5` child downweighting. Bad children receive less aggregation weight; adjusted child weights are normalized back to the same total child weight.

Because current GoCube search has no transpositions, child edge visits equal child visits and KataGo's edge-adjusted child weight specializes exactly to the child's aggregate `weightSum`.

## LCB

LCB is computed from child utility mean, utility-square mean and effective sample size, with KataGo's small variance prior. Root ending-score utility changes are added to the mean before the confidence comparison. Only children above `minVisitPropForLCB` relative to the non-LCB best are eligible. The best-LCB move receives the same bounded multiplicative play-selection-weight bonus used by pinned KataGo.

## Topology-specific substitution

KataGo's `interpolateEarly` scales elapsed halflives by `19 / sqrt(board_x * board_y)`. Cube and torus use `19 / sqrt(logical_point_count)`. No other formula is topology-specific in this block.

## Telemetry

Regular self-play records retain raw NN root policy, exploration-modified policy and raw visits, and now also expose pre-LCB and post-LCB play-selection weights, LCB values/radii, chosen LCB action, final normalized policy target, retrospectively removed exploration visits, and PASS-suppressed visits.
