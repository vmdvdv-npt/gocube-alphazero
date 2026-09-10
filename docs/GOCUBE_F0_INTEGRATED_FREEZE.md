# GoCube F0 integrated contract freeze

> Historical freeze note: this document preserves the exact F0 contract that
> was verified at the source anchor below. Its `komi=0.5` assertions are local
> reproducibility facts, not the current global GoCube/Torus komi policy. See
> `docs/KOMI_POLICY.md`; new development defaults to 0.5 but may explicitly
> research other finite values, while legacy 7.5 is forbidden in runtime.

Status: F0 complete on the integrated tree. This is a verification,
versioning, and documentation freeze; it does not claim V1 or V2 completion.

## Frozen source and stages

- integration base `main`: `efe31b7eefacbdd09abf4454a84ab66ffc1c9054`;
- integrated stages: S1 scoring/cleanup setup, S2 model contract, S3 runner
  termination, G1 structural features, H1 state/observation audit, M1 exact
  simple-ko, and verification groundwork;
- KataGo reference: `f6bc4b19a1686caa2d088b56251e8c11c8be6d51`;
- rules: version 3, local implementation version 5,
  `gocube-katago-rules-v3-implementation-v5`;
- search: `katago-pinned-search-v4`;
- replay: format v4, seven NN tensors plus row-aligned provenance sidecar and
  completion marker;
- termination: `gocube-termination-provenance-v1`;
- target provenance: `formal-runtime-result-provenance-v1`, encoded as
  `gocube-target-provenance-encoding-v1`;
- F0 experiment komi: `0.5`.

`katago-pinned-search-v4` is the integrated semantic identity. It freezes the
pinned reference, exact rule-derived M1 ko, root ending bonus, player-relative
value conversion, score utility, ownership interpretation, and the S3
runner/search-clone boundary. The protected budgets remain external:
ordinary self-play 50 simulations, fast self-play 20, Arena 50.

## Profiles

| Profile | Architecture | Observation | Structural channels | Action contract | Cube 4 topology |
| --- | --- | --- | ---: | --- | --- |
| `baseline` | `gocube-graph-v1` | `gocube-observation-v4-pass-would-end-phase`, `(18, 96, 1)` | 0 | 97 actions, `gocube-action-point-id-pass-v1` | unchanged |
| `g1` | `gocube-graph-structural-v1` | `gocube-observation-v5-structural-features`, `(20, 96, 1)` | 2 | 97 actions, `gocube-action-point-id-pass-v1` | unchanged |

Baseline and G1 share rules implementation and fingerprint, KataGo reference,
search/target/termination contracts, komi, topology, point order, adjacency,
action schema/size, and scorer semantics. Their permitted differences are
model profile, architecture identity/fingerprint, observation schema/shape,
and structural feature schema/channels. Cube 4 has 8 graph triangles and 24
triangle points; Torus 9 has no artificial cube-triangle features.

The machine-readable contract is
`tests/fixtures/gocube_f0_integrated_contract.json`. Its
`integration_base_commit` intentionally records the base tree rather than the
future commit containing the fixture itself.

## H1 post-merge probe

The same bounded corpus and semantic signature were run through both encoders:

| Profile | Samples | Observation groups | Collision groups | Semantic collisions | Bound |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline` | 267 | 263 | 3 | 0 | Cube 2 BFS depth 6, cap 256, plus phase/fork/cleanup samples |
| `g1` | 267 | 263 | 3 | 0 | same corpus and bound |

The three groups are expected aliases with equal semantic signatures. The
conservative finding is `no semantic collision found within bounded search`;
this is not a proof of full Markov sufficiency. Runtime episode termination is
audited separately because `episode_move_count` and `episode_move_limit` are
runner-owned and are not NN observation fields.

## Verification completed

The F0 cross-stage regressions cover:

- exact `v4` search version and fail-closed historical `v3` metadata;
- both profiles through checkpoint, manifest, catalog, loader, and
  `predict_for_search`;
- production MCTS conversion for black- and white-to-move player-relative
  values;
- S1 `5x5` scorer counterexample (`W-B = -3.5`, winner Black);
- M1 false/true exact-ko behavior and root effect;
- S3 formal pass, rule-cycle `NO_RESULT`, runtime force-score, and row-ordered
  provenance save/reload;
- Cube rotations and structural invariants, plus Cube 4/Torus 9 forward and
  MCTS smoke paths;
- protected `50/20/50` budgets and F0-local komi `0.5`.

No training experiment or B0/B1 strength claim is part of F0. V1 full
independent Cube verification is pending. V2 full product compatibility
verification is pending.
