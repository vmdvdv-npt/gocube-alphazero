# GoCube V1 difference registry

The entries below are stable adapter/scope IDs.  They explain representation
or applicability differences and are not permission to ignore a board, legal,
capture, ko, phase, terminal, winner, ownership, or score mismatch.

The machine-readable source is
`tests/reference/gocube_v1_difference_registry.json`.

| Difference ID | Upstream behavior | Local behavior | Reason/scope | Verification | Status |
| --- | --- | --- | --- | --- | --- |
| coordinate-encoding | KataGo uses rectangular `Loc`/`[x,y]` | GoCube uses canonical PointId and integer point actions | Coordinate normalization only | `T1` rectangular topology bridge | EXPLAINED_DIFFERENCE |
| action-indexing | KataGo PASS is `Board::PASS_LOC` | GoCube PASS is `Topology.point_count` | Action adapter only | `KatagoOracleProcess`, `v3_valid_moves` | EXPLAINED_DIFFERENCE |
| graph-neighbor-source | KataGo derives rectangle edges from board dimensions | GoCube reads `Topology.neighbor_indices` | Rectangular adapter vs closed-surface graph | native masks + independent graph fixtures | EXPLAINED_DIFFERENCE |
| score-sign | KataGo exposes white-minus-black final score | Training score head uses black-minus-white | Consumer-facing comparison normalization | `assert_snapshot_equal`, `normalized_score_target_v3` | EXPLAINED_DIFFERENCE |
| cube-topology-scope | Pinned KataGo Board is rectangular | Cube has six-face seams and vertex triangles | No direct upstream equivalent exists | Cube2–7 invariants, Cube4 graph corpus, rotations | EXPLAINED_DIFFERENCE |
| torus-topology-scope | Pinned KataGo Board has no wrap topology | Torus has horizontal/vertical wrap edges | No direct upstream equivalent exists | Torus9 wrap group/capture/ko corpus | EXPLAINED_DIFFERENCE |
| runtime-episode-policy | KataGo `maxMovesPerGame` is runner policy | GoCube `episode_move_limit` is runner policy | Runtime boundary is outside formal transition | S3 near-limit test; `apply_v3_action` remains formal-only | EXPLAINED_DIFFERENCE |

No production semantic mismatch is allowlisted.
