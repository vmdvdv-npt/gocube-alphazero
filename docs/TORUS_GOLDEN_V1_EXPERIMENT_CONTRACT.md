# Torus Golden graph-area v1 — Stage 0 experiment contract

Status: **normative Stage 0 passport**  
Profile ID: `gocube-torus-golden-graph-area-v1`  
Machine profile: `configs/gocube/torus_golden_v1.json`  
Contract schema: `1`  
Fingerprint algorithm: `sha256-canonical-json-v1`

This document and the JSON profile define the first end-to-end Torus proof. The JSON profile is the machine-readable effective contract; this document is its human-readable normative explanation. Any semantic change requires a new version/profile identity rather than an implicit default change.

## 1. Clean base and scope

The experiment is based on:

- base branch: `codex/torus-rebuild-v1`;
- base Git SHA: `c7a0fab2c708cbc9785f914ec6004fffeb7bb2a7`;
- task development must occur on a separate branch on top of that exact SHA;
- `main` is not the base of this architecture.

Stage 0 defines identities and semantics only. It does **not** implement a Golden referee, gameplay engine, scorer, Arena, MCTS rewrite, self-play, trainer, GraphNet redesign, batching, multiprocessing, Cython optimization, Cube support, fair-komi research, or UI/API integration.

Japanese-V3, legacy MCTS, historical Torus factory size restrictions, and checkpoint-supplied Arena defaults are not normative inputs to this profile.

## 2. Fingerprinted identities

The effective profile records these identities:

| Contract | Identity |
| --- | --- |
| Experiment | `sha256:1f5a1f821fc2e3dc52c957da9ffe1fc9313fb5b02bd81fd2f452233b57bd576e` |
| Topology | `sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef` |
| Rules | `sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842` |
| Observation | `sha256:d6e3aecc89f7df84f6e758423da4e3fe9269abeca644070db0be9261b30c6361` |
| Targets | `sha256:6dffad74c4f832741f9cd522aa407f3e88d6f6159ebe56c3dc226a52b4145e41` |

Fingerprints use canonical JSON (`sort_keys=true`, compact separators, ASCII encoding) followed by SHA-256. The topology fingerprint also includes the generated ordered adjacency table, so changing geometry or PointId/order changes identity. The rules fingerprint includes the topology fingerprint and explicit komi. The experiment fingerprint composes all Stage-0 semantic identities and execution/reproducibility policy.

A consumer must recompute and verify fingerprints before trusting this profile. Silent fallback, coercion, or accepting stale fingerprints is forbidden.

## 3. Topology contract — Torus 5×5

The first scientific test bed is a finite undirected Torus 5×5 graph:

- exactly 25 distinct vertices;
- `PointId = y * 5 + x`, row-major in `(y,x)` order;
- ordered neighbors are `[N,E,S,W]`;
- wrap-around is enabled independently on both coordinates;
- every vertex has exactly four **distinct** neighbors;
- no self-loops;
- no duplicate neighbors;
- every edge is reciprocal/undirected.

For `(x,y)`, the neighbors are `((x,y-1), (x+1,y), (x,y+1), (x-1,y)) mod 5`, converted to PointIds in the fixed order above.

The historical Torus `9/13/19` factory restriction is explicitly not part of the new contract. Smaller Torus 2×2/3×3 boards may later exist as test fixtures, but they do not define this experiment.

## 4. Rules contract — `graph-area-v1`

### 4.1 Placement

An ordinary move is defined in this exact order:

1. place the moving side's stone on an empty vertex;
2. find adjacent opponent groups with zero liberties after placement;
3. remove all such captured opponent groups;
4. evaluate the newly formed own group after captures;
5. if that own group has zero liberties, the move is illegal.

`suicide = forbidden` is fixed for this profile.

### 4.2 Positional superko

The ko rule is **positional superko**. An ordinary placement is illegal when its resulting stone arrangement reproduces any previous stone arrangement in the game's superko history.

The initial stone arrangement is inserted into that history. Superko history is part of the full rules/search game state. Therefore the same current board with different prior superko histories need not be the same full state.

The neural observation need not expose the entire history; that is a model approximation only. Rules and search must still use the complete history.

### 4.3 Pass and formal terminal condition

`PASS` is a distinct legal action in every nonterminal state.

- first PASS does not end the game;
- PASS increments consecutive-pass count;
- any ordinary legal move resets that count;
- two consecutive PASS actions end the game by rule.

There is no resign action in v1.

### 4.4 Explicitly absent v1 semantics

The following are not present:

- Japanese cleanup phases;
- encore;
- agreement removal;
- `whiteBonusScore`;
- Benson terminal shortcut;
- heuristic dead-stone removal;
- forced termination from NN evaluation.

## 5. Graph area scoring

At formal rule terminal, each color's area is:

`area = stones + owned empty points`

For each connected empty component:

- Black owns it iff its boundary contains Black and contains no White;
- White owns it iff its boundary contains White and contains no Black;
- otherwise it is neutral.

Stones remaining on the graph are never removed merely because a heuristic considers them dead.

The score is:

`margin_black = black_area - white_area - komi`

- `margin_black > 0` → `BLACK`;
- `margin_black < 0` → `WHITE`;
- `margin_black == 0` → `DRAW`.

## 6. Komi policy

The first proof profile pins **`komi = 0.5`**. This is the reproducible baseline for this experiment, not an architectural claim that every future research profile must use 0.5.

The general new-contract validator accepts explicitly provided finite alternative komi values for a future owner-approved version/profile. Any such change must be explicit, persisted, and identity-changing.

Komi must be:

- passed explicitly into future rules/search paths;
- included in rules and experiment identity;
- persisted in future manifests and checkpoints;
- validated as finite numeric data.

### 6.1 Forbidden legacy sentinel `7.5`

`7.5` is a legacy contamination sentinel for the new line. If it reaches any current runtime config, launcher, checkpoint metadata, manifest, migration, inherited default, or experiment profile, the path must **fail closed**. The diagnostic must identify likely legacy/stale-artifact contamination and instruct the operator to contact the project owner.

It is forbidden to:

- use 7.5 as an ordinary research value;
- silently continue with 7.5;
- automatically rewrite `7.5 → 0.5`.

The project-wide `validate_gocube_komi` policy is the Stage-0 validation primitive for this rule.

## 7. Rule results vs technical termination

Rule-level completed-game outcomes are exactly:

- `BLACK`;
- `WHITE`;
- `DRAW`.

Execution statuses are separate:

- `TRUNCATED_MOVE_LIMIT`;
- `ERROR`.

Technical statuses are not `DRAW`, `LOSS`, or `NO_RESULT`, and must not produce a WDL training target or enter training replay in v1.

There is no rule-level `NO_RESULT` in this v1 contract. Introducing one later requires a new versioned target contract.

## 8. Execution watchdog

The first Torus 5×5 execution watchdog is:

`20 × number_of_points = 500 actions`

PASS counts as an action. The watchdog is an execution safety cap, **not a game rule** and not a winner adjudicator.

If the second consecutive PASS occurs on the final allowed action, formal terminal semantics are applied first; the game is scored rather than marked truncated.

If more than 5% of self-play attempts later reach `TRUNCATED_MOVE_LIMIT`, a long learning run must not begin until the cause is investigated. The 5% value is an engineering warning threshold, not a mathematical property of the game.

## 9. Value target contract — WDL side-to-move v1

Contract ID: `gocube-wdl-side-to-move-v1`.

The value vector is:

`[WIN, DRAW, LOSS]`

and is always relative to the **side to move at that replay position**.

Utility is:

`u = P(WIN) - P(LOSS)`

Normative `z` construction rules for the later replay builder:

- derive `z` only from a rule-level terminal outcome;
- for a BLACK terminal winner, encode WIN at Black-to-move replay positions and LOSS at White-to-move positions;
- for a WHITE terminal winner, encode WIN at White-to-move replay positions and LOSS at Black-to-move positions;
- DRAW always maps to `[0,1,0]`;
- technical termination produces no WDL target.

Legacy `[WIN, LOSS, NO_RESULT]` is incompatible even though its tensor length is also three. Compatibility is semantic/fingerprint based, never shape based.

## 10. Policy target contract

The first baseline policy target is normalized root MCTS visit count:

`π(a) = N(a) / Σ N(legal actions)`

Requirements:

- every illegal action has zero mass;
- legal-action target mass sums to one;
- root visit sum must be positive;
- fast-search target generation is absent;
- forced-playout pruning is absent;
- LCB target transformation is absent.

This section fixes the target semantics only; Stage 0 does not implement MCTS or the target builder.

## 11. Observation contract v1

Schema ID: `gocube-torus-golden-observation-v1`, version `1`.

The point-axis length is 25 and the minimum point channels are semantically fixed:

| Index | Channel | Semantics |
| ---: | --- | --- |
| 0 | `own_stones` | 1 on stones belonging to the side to move, else 0 |
| 1 | `opponent_stones` | 1 on opponent stones, else 0 |
| 2 | `side_to_move_color` | constant plane, +1 Black / -1 White |
| 3 | `previous_pass` | constant plane, 1 iff immediately previous action was PASS |
| 4 | `legal_point_mask` | 1 iff the PointId placement is legal under the full state |
| 5 | `komi` | constant plane with explicit game komi |

The full legal-action mask has length 26: PointIds `0..24`, then PASS at index `25`.

The NN observation intentionally does not encode complete superko history. That does not weaken rules/search state: legality and search must retain and consult full superko history.

## 12. Self-play search and Golden Arena search are different contracts

Two identities are reserved now so later implementation cannot accidentally reuse one settings object.

### Self-play placeholder

`gocube-torus-golden-self-play-search-v1-placeholder` may later define its own Dirichlet root noise, move temperature, exploration policy, and simulation budget. Those values are not implemented or chosen in Stage 0.

### Golden Arena placeholder

`gocube-torus-golden-arena-search-v1-placeholder` fixes these semantic constraints:

- root noise OFF;
- fast search OFF;
- move temperature `0`;
- root policy temperature OFF;
- resign OFF;
- identical search budget for both models;
- sims/cpuct/FPU come from the Arena contract, never silently from checkpoint metadata;
- no inheritance from self-play config.

Specific sims/cpuct/FPU numbers remain deliberately unset in Stage 0 and must be versioned when Stage 1+ chooses them.

## 13. Future checkpoint semantic identity

A future checkpoint for this line must persist at least:

- rules profile ID and fingerprint;
- topology fingerprint;
- board size;
- PointId/order identity;
- komi;
- observation schema ID/version/fingerprint;
- target contract ID/version/fingerprint;
- value-head semantics;
- network heads and shapes;
- parent/source run identity;
- model hash.

Any required metadata mismatch must fail closed at load. There is no implicit fallback, inherited default, or semantic coercion.

## 14. Reproducibility passport

The first experiment identity includes:

- experiment ID and fingerprint;
- exact base branch/SHA;
- rules profile/fingerprint;
- topology/fingerprint and PointId order;
- explicit komi;
- observation schema/version/fingerprint;
- target contract/version/fingerprint;
- current network interface contract where it is already knowable;
- separate self-play and Golden Arena placeholder identities;
- execution watchdog;
- RNG seed policy;
- artifact layout.

Future stochastic execution must persist an explicit master seed and deterministically derived stream seeds.

Progress must later be reported through real work counters:

- completed games;
- valid replay positions;
- train samples;
- optimizer updates;
- NN evaluations.

`iteration=N` alone is not a sufficient progress measurement.

## 15. Artifact layout

Stage-0 normative artifacts:

- `configs/gocube/torus_golden_v1.json` — effective machine contract;
- `docs/TORUS_GOLDEN_V1_EXPERIMENT_CONTRACT.md` — human normative contract;
- `alphazero/envs/gocube/torus_golden_contract.py` — small loader/validator/fingerprint helper;
- `tests/test_torus_golden_v1_contract.py` — targeted contract regression tests.

Future run artifacts are reserved beneath `runs/<run_id>/` with `manifest.json` and `checkpoints/`, but Stage 0 does not create a runner.

## 16. Stage boundary

The next stage is **Stage 1 — Golden Torus Rules Core Minimal**. Stage 1 may implement the rule mechanics defined here, but must consume this profile explicitly and must not reinterpret legacy Japanese-V3 behavior as the new contract.
