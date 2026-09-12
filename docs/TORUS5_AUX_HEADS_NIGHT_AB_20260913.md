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

## Completed run

`torus5-aux-heads-night-ab-20260913` completed **PASS** from compute source
commit `1643cabe4244f40f849fbc94de2864c586ede871`, using 16 workers and
`komi=0.5`. All self-play, diagnostic Arena, primary Arena, progression, and
same-model control technical counts were zero.

### Final M2 primary ranking

| Rank | Variant | Combined mean score | Games | Seed A | Seed B |
|---:|---|---:|---:|---:|---:|
| 1 | WDL + ownership | 0.5469 | 768 | 0.5313 | 0.5625 |
| 2 | WDL + ownership + score | 0.5169 | 768 | 0.5625 | 0.4714 |
| 3 | WDL | 0.4987 | 768 | 0.4375 | 0.5599 |
| 4 | WDL + score | 0.4375 | 768 | 0.4688 | 0.4063 |

Training baseline verdict: **INCONCLUSIVE**. Ownership is the combined leader,
but the joint/score arms show seed reversal and the intervals are not decisive;
the data does not justify promoting a single configuration as a proven Torus
9x9 baseline. The practical candidate for a follow-up is WDL + ownership, not
a scientific CLEAR WINNER.

### FIRST-MOVE ADVANTAGE

**FIRST-MOVE ADVANTAGE: DETECTED** under the declared stochastic same-model
self-play estimate. Across 8 final M2 controls and 2048 valid games, Black
won 1178 (57.52%), with Wilson 95% CI **[55.37%, 59.64%]**. Seed A was
55.96% (CI [52.90%, 58.97%]); Seed B was 59.08% (CI [56.04%, 62.05%]); all
eight model point estimates were above 50%. Mean raw Black area advantage was
**+1.3672** points, and mean final Black margin after komi 0.5 was
**+0.8672**. The result is therefore a detected Black/first-player bias for
this protocol, not evidence for changing komi.

Canonical Stage4 self-play was 59.18% Black wins; new I1/I2 exploratory
buckets were retained separately in the machine report. Total scientific
counts were **3328 self-play games**, **2304 Arena games**, and **2048 final
control games**; all reported technical counts were zero. Wall time was
2463.9 seconds, peak parent RSS about 980.7 MB.
