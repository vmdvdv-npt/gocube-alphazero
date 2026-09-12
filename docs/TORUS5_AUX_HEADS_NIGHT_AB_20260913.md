# Torus 5x5 auxiliary-head night A/B protocol

This experiment is fixed to the proven Golden Torus 5x5 contract and
`komi=0.5`. It compares exactly four arms:

1. `wdl`
2. `wdl+ownership`
3. `wdl+score`
4. `wdl+ownership+score`

The executable protocol is `tools/torus_aux_heads_night_ab.py`. It performs
fail-fast target, shared-M0, tiny training, tiny self-play, checkpoint, and
process-parallel Arena checks before the full run. Run it with the repository
environment:

```bash
.venv/bin/python tools/torus_aux_heads_night_ab.py \
  --run-id torus5-aux-heads-night-ab-20260913 --workers 16
```

## Frozen design

Each of two independent seeds creates one shared M0 initialization. The four
arms use bit-identical policy/WDL parameters before the first optimizer step;
auxiliary parameters are deterministic arm-local initialization. Iteration 1
uses one immutable 128-game M0 self-play corpus per seed for all four M1
checkpoints. Iteration 2 uses a fresh independent 128-game corpus for each
M1 arm, then trains that arm to M2. Stage4's fixed batch size (64), Adam,
learning rate, weight decay, one-pass deterministic sample ordering, and 1:1
new-position sample budget are retained.

Ownership targets are the exact final Golden referee ownership classes in the
sample side-to-move perspective: `OWN`, `OPPONENT`, `NEUTRAL`. Score targets
are the exact final Golden margin in the same perspective. Training divides
score by fixed `25.5`; diagnostics and replay retain the exact margin, so the
normalization is reversible. Technical games are rejected from replay and
statistics.

The primary comparison is the six-way M2 round-robin on the existing frozen
64-pair / 128-game color-swapped evaluation corpus for each seed. Progression
checks use a fixed diagnostic subset. Final same-model controls use 256 fresh
games for each of the eight M2 checkpoints under the proven stochastic
self-play protocol and are labelled `self-play first-move estimate`.

The complete machine-readable evidence is written to the run directory under
`runs/torus5-aux-heads/` (ignored runtime artifacts). The final committed
machine and human reports are copied to:

* `docs/TORUS5_AUX_HEADS_NIGHT_AB_20260913.json`
* this file, with the result appended after the run

The runner fails closed on non-0.5 komi, technical Arena/self-play games,
target/source drift, checkpoint identity drift, or incomplete arms. It never
converts a failed technical arm into a draw.
