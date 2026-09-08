# GoCube V1 verification report

This report records the local V1 gate from the branch
`codex/v1-independent-rule-verification`, based on merged PR55/F0:
`25b9f7f063acd0fa1921ab8f3c5a2adb8e8c8bad`.

## Git

| Item | Value |
| --- | --- |
| F0 base SHA | `25b9f7f063acd0fa1921ab8f3c5a2adb8e8c8bad` |
| Branch | `codex/v1-independent-rule-verification` |
| V1 implementation commit | `f74fe33ac52a6b1ddc03ec07764be89458e03279` |
| Final branch HEAD | reported by the final task handoff after this report commit |

## KataGo

| Item | Result |
| --- | --- |
| Pinned commit | `f6bc4b19a1686caa2d088b56251e8c11c8be6d51` |
| Rules | V3 / SIMPLE ko / TERRITORY / SEKI tax / no multi-stone suicide / komi 0.5 |
| Native oracle build | PASS: `tools/katago_reference/build_oracle.sh` |
| Static differential cases | 28 |
| Generated sequences | 8 |
| Generated compared steps | 204 |
| Legal-mask/snapshot mismatches | 0 |

Generated sequences use fixed seeds `20260901` through `20260908`, lengths
`12,24,48,12,24,48,12,24`, and generator
`gocube-v1-legal-intersection-v1`.  Every action was selected only after the
two masks were asserted equal; every post-action semantic snapshot was then
compared.  All eight runs reached their configured bound without an early
terminal.

## Cube and Torus

The Cube source corpus contains 25 typed fixtures, with 22 rotation-safe
source fixtures and 528 metamorphic rotation variants.  Cube2–Cube7 topology
invariants pass; Cube4 has 96 points, eight graph triangles, and 24 triangle
points.  Mandatory Cube graph families (groups, liberties, seams, vertices,
captures, ko, eyes, false-eye, seki/dame and intruder) have independent graph
evidence and production comparisons.

The Torus9 source corpus contains four fixtures: wrap group, wrap capture,
wrap ko, and no-Cube-triangle topology control.  Torus9, Torus13, and Torus19
degree/wrap invariants pass.  Cube-specific triangle features remain zero on
Torus.

## Solver

The bounded solver gate includes one proved tiny continuation.  The separate
resource-exhaustion regression intentionally returns `unknown`; it is a
fail-safe test and is not part of the mandatory V1 acceptance corpus.  No
mandatory fixture has solver status `unknown`.

## Differences and matrix

The complete registry is in
`tests/reference/gocube_v1_difference_registry.json`; all seven entries are
stable `EXPLAINED_DIFFERENCE` adapter/scope records.  There are no unresolved
production rule differences.  The canonical family matrix is in
`docs/GOCUBE_V1_VERIFICATION_MATRIX.md` and all mandatory rows are `PASS`.

## Protected scope

Production rules, search, training, network architecture, Cython sources, and
the TypeScript product were not changed by V1.  The protected settings remain
komi `0.5`, ordinary self-play `50`, fast self-play `20`, and Arena `50`.
`episode_move_limit` remains runtime policy and is not part of formal rules
differential.
