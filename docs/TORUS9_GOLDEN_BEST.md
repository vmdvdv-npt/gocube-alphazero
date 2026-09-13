# CURRENT TORUS9 GOLDEN BEST

Status: **CURRENT_BEST**

Run/checkpoint: `torus9-stable-learning-20260913-v1 / M8`

Model hash: `sha256:0c2beca91354b04b26f7092c94b5c60b217c5db2d5e8cb5403a124bb39738761`

Checkpoint artifact SHA-256: `sha256:a0e54de1e4370979d8447867113c9594817638dab3816634ef3722e6103529d0`

Artifact source commit: `25dc804cced364c4289e4b0580e3799c4437e288`; PR #86 merge anchor: `88b1803cf179c6fa93f2e9610963eeed931d09b1`.

Architecture: `GoldenGraphNetV2-Torus9-8Block`, hidden 64, 8 blocks, policy `[82]`, WDL `[3]`, komi `0.5`.

Training: 8 × 64 self-play games; rolling replay 3 generations / 20,000 positions; Adam `lr=0.001`, `wd=0`, 80 optimizer steps × batch 64 per iteration.

Arena evidence: NEW M8 vs OLD M8 `59 / 3 / 0`; M8 vs M1 `128 / 0 / 0`; M8 vs M0 `119 / 7 / 0`; M8 vs M4 `92 / 24 / 0`. Technical games are stored separately and excluded from W/L/D.
