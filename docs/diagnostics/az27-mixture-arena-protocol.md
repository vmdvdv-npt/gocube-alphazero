# AZ27: mixture B versus AZ23 B/M269

Predeclared comparison: final B snapshot at 640 updates from completed
az23-r2-policy-target-mixture-20261012-v1 against its exact registered parent.
No snapshot selection, training, self-play, or extra comparisons in this job.

- 384 games, 192 paired starts with color swap; 64 simulations, tree reuse on.
- Master seed 2026101210; standard Torus9 komi 1.5 Arena settings.
- Primary criterion: valid complete Arena, candidate win fraction > 0.5 and
  two-sided exact paired sign-test p < 0.05 (decisive pair score differences;
  tied pair scores excluded). Report raw W/L/D, pair outcomes and interval.
- A failed criterion means no established advantage, not equivalence.
- No automatic promotion, extension of game count, or follow-up training.
- Operator requested launch only, without subsequent assistant monitoring.

Use configs/operator/az27-register-mixture-b-20261012.json through the public
register-diagnostic-checkpoint command, then job --check and job with
configs/operator/az27-mixture-b-arena-20261012.json. Registration preserves source
bytes and creates only a new evaluation identity, not a training lineage.
