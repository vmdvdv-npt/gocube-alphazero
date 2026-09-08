# GoCube V1 verification matrix

Status: accepted V1 corpus.  Base: merged PR55/F0,
`25b9f7f063acd0fa1921ab8f3c5a2adb8e8c8bad`.

Every mandatory row is `PASS`; there are no `PENDING`, `UNKNOWN`,
`STRUCTURAL_ONLY`, or `UNRESOLVED` rows.  The source fixture is counted once;
24 Cube rotations are metamorphic variants within the same provenance family.

| Family | Fixture count | Independent oracle | KataGo applicable? | Production compared? | Rotations | Status | Difference ID | Notes |
| --- | ---: | --- | :---: | :---: | --- | --- | --- | --- |
| rectangular basic placement | 28 | pinned KataGo | yes | yes | n/a | PASS | none | Static pinned corpus |
| rectangular capture | 28 | pinned KataGo | yes | yes | n/a | PASS | none | Removed points and counts |
| suicide | 28 | pinned KataGo + graph control | yes | yes | n/a | PASS | none | Includes capture-before-suicide |
| true ko | 1 | pinned KataGo + independent restoration | yes | yes | 24 where Cube-safe | PASS | none | Immediate recapture forbidden |
| false ko | 1 | independent restoration | no | yes | 24 | PASS | none | Apparent recapture is suicide |
| PASS | 28 | pinned KataGo + independent PASS | yes | yes | n/a | PASS | none | Board preserved, turn consumed |
| MAIN→C1 | 1 | pinned KataGo | yes | yes | n/a | PASS | none | Exact two-ending-pass trigger |
| C1→C2 | 1 | pinned KataGo | yes | yes | history-aware | PASS | none | Start colors captured |
| C2→score | 1 | pinned KataGo | yes | yes | history-aware | PASS | none | Formal pass terminal |
| cleanup ko | 4 | pinned KataGo | yes | yes | history-aware | PASS | none | Block/release/repeat |
| cycle | 1 | pinned KataGo | yes | yes | history-aware | PASS | none | Rule `NO_RESULT`, not runtime cap |
| pass-alive | 2 | pinned KataGo + independent vital-region/placement proof | yes | yes | Cube/Torus smoke | PASS | none | Positive two-vital-region proof plus closed-board placement exhaustion |
| vertex groups | 4 | independent graph | no | yes | 24 per source | PASS | none | Cube4 vertex triangles |
| vertex capture | 3 | independent graph | no | yes | 24 per source | PASS | none | Includes vertex and multi-group capture |
| seam groups | 1 | independent graph | no | yes | 24 per source | PASS | none | Cross-face adjacency |
| seam capture | 1 | independent graph | no | yes | 24 per source | PASS | none | Capture through seam |
| global connectivity | 2 | independent graph | no | yes | 24 per source | PASS | none | Three-face path and cut |
| eyes | 2 | independent graph + reviewed proof | no | yes | 24 per source | PASS | none | Obvious and vertex-related true eyes |
| false eye | 1 | independent graph | no | yes | 24 per source | PASS | none | Mixed-border control |
| seki | 1 | independent bounded exhaustive continuation + KataGo seki-tax analog | no | yes | 24 per source | PASS | none | Settled two-shared-liberty seki; all first placements and defensive replies enumerated |
| dame | 1 | independent mixed-border region proof | no | yes | 24 per source | PASS | none | Neutral region derived before production comparison |
| scoring/setup | 4 | pinned KataGo | yes | yes | not multiplied | PASS | none | Includes S1 W−B = −3.5 |
| S1 intruder | 1 | independent graph + pinned analog | no | yes | history-aware | PASS | none | Cube graph, no fake rectangle oracle |
| ownership | 4 | pinned KataGo + graph regions | yes | yes | invariant | PASS | none | Black/White/Neutral and masks |
| Torus wrap group | 1 | independent graph | no | yes | Torus smoke | PASS | none | Horizontal wrap |
| Torus capture | 1 | independent graph | no | yes | Torus smoke | PASS | none | Horizontal wrap capture |
| Torus ko | 1 | independent graph | no | yes | Torus smoke | PASS | none | Horizontal wrap restoration |
| rotations | 22 source fixtures | metamorphic + independent source | no | yes | 24 per source | PASS | none | Source fixture split preserved |

## Differential accounting

The CI-sized generated corpus uses generator
`gocube-v1-legal-intersection-v1`, seeds
`20260901, 20260902, 20260903, 20260904, 20260905, 20260906, 20260907,
20260908`, and requested lengths `12, 24, 48, 12, 24, 48, 12, 24`.
Each pre-action legal mask and each accepted post-action semantic snapshot is
compared.  Reaching a configured bound after matching all steps is a passing
bounded run.

The native static corpus contains 28 fixtures, all pinned to KataGo commit
`f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.  The runner compares board,
current player, legal actions, captures, phase, PASS semantics, ko blocks,
terminal/no-result state, winner, score, formal area, ownership and score
offset where applicable.

## Difference registry

The complete machine-readable registry is
`tests/reference/gocube_v1_difference_registry.json`.  All entries are
adapter/scope differences, not unexplained rule mismatches.  There are zero
unresolved semantic differences.

## V2 handoff

Only `verified` V1 source fixtures are eligible for the existing product
boundary exporter.  `training_internal_cleanup` remains a separate section;
V1 does not compare against the real TypeScript `GameEngine`.
