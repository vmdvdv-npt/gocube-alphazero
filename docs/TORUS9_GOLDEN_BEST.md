# CURRENT TORUS9 GOLDEN / `new_komi` TRANSITION

Status: **M137 REFERENCE FROZEN · `new_komi` PREPARED FOR FRESH 5CH KOMI CALIBRATION**

## Reference checkpoint

Canonical parent lineage: `torus9-m125-continuous-v2-gen6-20260922-v1`

Canonical parent checkpoint: `M137`

Checkpoint artifact SHA-256:

`71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe`

The historical M137 network is `GoldenGraphNetV2-Torus9`, hidden 80, 8 blocks,
policy `[82]`, WDL `[3]`, ownership `[81,3]`, score `[1]`.

## M137 6CH -> 5CH conversion

PR #198 established the 5-channel representation:

`own stones · opponent stones · side-to-move color · previous pass · legal mask`

The persistent komi input is removed. For the M137 model trained with fixed
komi `0.5`, the first affine layer is folded as:

`W5 = W[:,0:5]`

`b5 = b + 0.5 * W[:,5]`

All remaining model parameters are copied unchanged.

Converted M137 model hash:

`sha256:f4fc0e173ed9cea1a6274bb613a95cb6ba428461f87eb60313deb390261927bc`

PR #198 parity evidence: policy max abs diff `9.536743e-06`, WDL
`3.814697e-06`, ownership `3.397465e-06`, score `1.341105e-07`;
selected MCTS action parity `256/256`. This is an inference/model conversion
result, not proof that future training trajectories are identical.

## `new_komi` lineage

Lineage ID: **`new_komi`**

Bootstrap model: M137-5CH derived from the canonical M137 checkpoint above.

The parent M137 checkpoint is referenced by lineage/checkpoint/SHA identity.
It is not copied.

### Adam migration

The bootstrap keeps the M137 Adam state wherever the 6CH -> 5CH mapping is
unambiguous:

- all unchanged trunk/head parameter states are copied exactly;
- `input_projection.weight` Adam tensor states are cropped from 6 columns to
  the first 5 columns;
- `input_projection.bias` is the folded parameter `b + 0.5*W6`, so no exact
  merged Adam moment history exists. Only this parameter's first/second
  moments are reset to zero;
- the global Adam step for the folded bias is retained, so the optimizer clock
  stays continuous with the migrated parameters.

This conversion is explicitly recorded as
`adam-preserve-exact-crop-input-reset-folded-bias-moments-v1`.

### Replay/history boundary

`new_komi` starts with **no inherited replay history**.

No M137 parent replay, rolling replay, self-play rows, or replay references are
carried into the lineage. The new line may accumulate only self-play generated
after its own scientific komi contract is selected.

The parent checkpoint remains a provenance dependency only.

### Komi and training status

The source M137 was trained under actual komi `0.5`.

`new_komi` does **not** silently choose a replacement komi. Its bootstrap
status is:

`BLOCKED_PENDING_KOMI_CALIBRATION`

The bootstrap checkpoint is **optimizer-ready but not yet production-training-ready**.
No immutable `new_komi` training configuration is published before calibration,
so selecting the actual rules komi does not require mutating an already-pinned
scientific training contract.

A fresh 5-channel calibration must select the real rules komi first. After
selection, the production training binding must consume this bootstrap and
self-play, WDL/score targets, Arena, and all new replay data must use that same
actual komi.

`7.5` is legacy/error and must not be used.

## Storage

Active lineage location:

`runs/torus9/active/new_komi/`

The lineage owns its derived 5CH bootstrap checkpoint and future artifacts.
The canonical M137 source remains in its original lineage and is referenced by
identity. No parent checkpoint or replay dataset is duplicated.
