SYMMETRY AUGMENTATION: REDUNDANT

# Torus9 symmetry-equivariance audit

This report is diagnostic-only. No self-play, two-iteration A/B, or Arena games were started.

## Scope and provenance

- Frozen checkpoint: `/home/codex/projects/gocube-alphazero/runs/torus9-alpha-score-ab/torus9-alpha-score-ab-20260913-v1/arms/B/checkpoints/M12-B.pt` (M12-B, `sha256:2ba14fc38a7944dbe1d505f849e02a7acb18f0386518a07c2e03f6320b8df3ae`).
- Immutable corpus: `/home/codex/projects/gocube-alphazero/runs/torus9-alpha-score-ab/torus9-alpha-score-ab-20260913-v1/selfplay/D12-B/replay.jsonl` (5987 replay rows).
- Selected real states: 8; selected training batch: 64 rows.
- Parent semantics: Torus 9x9, WDL + ownership, komi 0.5, alpha 0.11, score OFF, 8 blocks / hidden 64.

## 1. Training pipeline

Current Torus9 training has no geometric data augmentation; the generic symmetricSamples name is not a Torus9 symmetry implementation.

The generic `symmetricSamples` spelling is not geometric augmentation here: the production Golden Torus9 trainer reads raw observation/target rows. The unrelated legacy/Cube/5x5 diagnostic utilities are not on this Torus9 path.

## 2. Group and topology

Generated and checked `648` unique transformations = D4 x Z9 x Z9 = 8 x 9 x 9.
Checked `52488` point-to-neighborhood mappings; failures: `0`.

## 3. State-level correctness

Checked `5184` transformed states, `5184` legal masks, `2460` mapped legal transitions, and `5184` terminal score/winner/ownership results.

The state transform carries the current board and every positional-superko history board. Side to move, pass count, komi, rules identity, and PASS semantics remain scalar/invariant. The network observation still does not encode the full history; that information boundary is separate from symmetry correctness.

## 4. Frozen M12-B inference

Errors below are absolute errors after inverse point-action/ownership permutation. Raw logits are compared before softmax, and probabilities are compared after softmax.

| Family | Head | max | mean | p99 |
|---|---|---:|---:|---:|
| d4 | policy_logits | 1.907e-06 | 1.401e-07 | 9.537e-07 |
| d4 | policy_probabilities | 1.192e-07 | 1.352e-09 | 1.118e-08 |
| d4 | wdl_logits | 4.768e-07 | 1.001e-07 | 4.768e-07 |
| d4 | wdl_probabilities | 1.788e-07 | 1.890e-08 | 1.290e-07 |
| d4 | ownership_logits | 7.153e-07 | 6.719e-08 | 4.768e-07 |
| d4 | ownership_probabilities | 2.384e-07 | 1.219e-08 | 1.192e-07 |
| translations | policy_logits | 9.537e-07 | 1.416e-09 | 0.000e+00 |
| translations | policy_probabilities | 2.384e-07 | 6.421e-10 | 5.588e-09 |
| translations | wdl_logits | 9.537e-07 | 9.217e-08 | 4.768e-07 |
| translations | wdl_probabilities | 1.490e-07 | 1.552e-08 | 1.192e-07 |
| translations | ownership_logits | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| translations | ownership_probabilities | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| combined | policy_logits | 1.907e-06 | 1.402e-07 | 9.537e-07 |
| combined | policy_probabilities | 2.384e-07 | 1.349e-09 | 1.118e-08 |
| combined | wdl_logits | 9.537e-07 | 9.182e-08 | 4.768e-07 |
| combined | wdl_probabilities | 1.788e-07 | 1.459e-08 | 1.192e-07 |
| combined | ownership_logits | 7.153e-07 | 6.719e-08 | 4.768e-07 |
| combined | ownership_probabilities | 2.384e-07 | 1.219e-08 | 1.192e-07 |

## 5. Training-step equivalence

Model A and Model B were loaded independently from the same M12-B checkpoint and Adam state. A used the original batch; B used the same 64 rows with a deterministic mix of rotation, reflection, translation, and combined D4+translation transforms. WDL targets stayed unchanged.

Loss absolute differences: `{"ownership_loss": 5.960464477539063e-08, "policy_loss": 0.0, "total_loss": 0.0, "wdl_loss": 0.0}`.
Gradient error max/mean/p99: `1.416e-07` / `2.938e-10` / `2.328e-09`.
Parameter-delta error max/mean/p99: `1.490e-08` / `2.281e-11` / `1.164e-10`.
Adam-state error max/mean/p99: `1.397e-08` / `1.471e-11` / `2.328e-10` across `240` tensor entries.

## Decision

Topology, full state transformation, frozen inference, and one-step Adam training equivalence all passed within the declared float tolerance.

Explicit symmetry augmentation is redundant for this Torus9 GraphNet. The architecture is equivariant to the stronger spatial automorphism group D4 x Z9 x Z9 (648 transformations), so transformed rows reproduce the same optimizer signal up to floating-point noise.

No conditional A/B was run, as required.

## Artifacts

- `audit.json`: complete machine-readable result.
- `state-transform-manifest.json`: selected states and all 648 permutations.
- `audit.md`: this report.
