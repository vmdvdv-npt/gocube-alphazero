# Torus9 M137 5-channel parity report

Source: canonical lineage `torus9-m125-continuous-v2-gen6-20260922-v1`, checkpoint `M137`.

- Source checkpoint SHA-256: `sha256:71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe`
- Source model hash: `sha256:00bfb3bcec6ae1e2144ef1951d44ee64ea0fd8955133f444fd9e0168667ada07`
- Source architecture: `GoldenGraphNetV2-Torus9`, hidden 80, blocks 8, observation `[6,81]`
- Target architecture: `GoldenGraphNetV2-Torus9-M137-5CH`, observation `[5,81]`
- Converted model hash: `sha256:f4fc0e173ed9cea1a6274bb613a95cb6ba428461f87eb60313deb390261927bc`
- Converter commit: `2f80cd06815a40c324e9c6cbbb92210fd7b77af9`

The komi channel was removed with:

```text
W5 = W[:, 0:5]
b5 = b + 0.5 * W[:, 5]
all remaining parameters copied without modification
```

The derived checkpoint is model/inference-only: `optimizer_conversion = not_performed`,
`training_ready = false`. It references the canonical M137 parent and does not copy
M137 or replay data. The Run Storage artifact is:
`runs/torus9/active/torus9-m137-5ch-derived-20260925-v1/`.

## Numerical parity

The harness used the first deterministic bounded 4096 rows from the canonical
`iter-137-fresh.jsonl`; it did not scan the full replay. All compared values were
finite (`0` non-finite values per tensor). Errors are absolute FP32 output errors;
the folded input projection uses FP64 accumulation for this inference-only boundary
and returns FP32 activations.

| Tensor | Max abs error | Mean abs error | P99 abs error |
| --- | ---: | ---: | ---: |
| First projection | `1.1920928955078125e-07` | `1.1242430222982539e-08` | `5.960464477539063e-08` |
| Encoded nodes | `5.245208740234375e-06` | `7.003628696796893e-08` | `5.960464477539062e-07` |
| Policy logits | `9.5367431640625e-06` | `6.030430352150621e-07` | `2.86102294921875e-06` |
| WDL logits | `3.814697265625e-06` | `2.659714179268728e-07` | `1.9073486328125e-06` |
| Ownership logits | `3.3974647521972656e-06` | `1.3688315762628018e-07` | `7.152557373046875e-07` |
| Score output | `1.341104507446289e-07` | `1.730040821712464e-08` | `7.450580596923828e-08` |

All maxima are at or below the `1e-5` acceptance threshold.

## Behavioral parity

The bounded diagnostic used 256 real positions, 64 deterministic PUCT simulations
per position, noise off, temperature 0, fast search off, resign off, and identical
cpuct/FPU settings.

- Selected-action parity: `256/256`
- Common root-Q maximum absolute error: `8.344650268554688e-07`
- Exact root-visit distributions: `253/256`
- Audited root-visit differences: positions `1` and `125` had L1 delta `2`; position
  `208` had L1 delta `4`; each per-edge delta was at most `1`.
- No selected action differed. The three visit-allocation differences are retained in
  the machine-readable parity report as tie-sensitive/symmetric root diagnostics.

## Actual-komi independence

For the same board position, target observations for komi `0.5`, `2.5`, and `4.5`
were identical, and neural outputs were identical (`max abs error = 0`). Referee
terminal margins remained komi-dependent: `-0.5`, `-2.5`, and `-4.5`.

Final result: **PARITY PASS**
