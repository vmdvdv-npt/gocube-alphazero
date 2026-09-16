# Torus 9x9 WDL + ownership A/B

> Historical / retired evaluation evidence. Do not use the Arena settings in
> this report as a current Torus 9×9 entrypoint; current strength comparisons
> use the Golden standard-64 contract, while high-volume settings are
> performance-only.

This controlled experiment starts from the merged PR88 tree and the stable
`torus9-stable-learning-20260913-v1 / M8` checkpoint. It compares two branches
for exactly two fixed training rounds at `komi=0.5`:

- A: WDL-only objective; the common ownership head is present but its loss
  weight is `0.0`.
- B: WDL + ownership objective; the same ownership head uses the existing
  Golden ownership contract and unit loss weight.

The ownership targets are produced by the existing
`golden-referee-final-state-v1` implementation in the sample's side-to-move
perspective, with classes `OWN`, `OPPONENT`, and `NEUTRAL`. Both branches use
the same optimizer state continuation from M8 step 640, the same deterministic
80 x batch-64 budget per round, and the same rolling replay contract.

## Shared data and evaluation

D9 and D10 were each generated once from frozen M8 and reused by both arms:

| Corpus | Games | Positions | Seed | Replay fingerprint |
|---|---:|---:|---:|---|
| D9 | 64 | 6039 | 2026091412 | `sha256:fa2b60acdd5e374165226af0692c8cd665da774280ccf2975a1f40f370da6261` |
| D10 | 64 | 6669 | 2026091413 | `sha256:852734f23a75962b510df0b56dc48937357338c29edef5c3f361dd9380cb417a` |

No M9 Arena was run. After both branches reached M10, the only Arena was
`M10-B vs M10-A`: 64 games, 32 paired starts with color swap, 16 workers,
batched inference (`arena_batch_size=8`, wait `6 ms`), 64 simulations, noise
OFF, temperature OFF, fast OFF, and watchdog 1000. The observed mean inference
batch size was **17.536 rows**, passing the minimum-16 performance guard. There
were no technical Arena games; technical outcomes are never mapped to W/L/D.

## Result

| Comparison | W/L/D | Valid pairs | Technical games | Mean pair score | Verdict |
|---|---:|---:|---:|---:|---|
| M10-B vs M10-A | 44 / 20 / 0 | 32 | 0 | 0.6875 | **B HAS USEFUL EVIDENCE** |

This is useful evidence for ownership under this fixed protocol, not a
canonical Golden promotion or a formal universal statistical threshold. No
adaptive Arena extension was performed.

Complete machine-readable evidence is in the ignored runtime artifact
`runs/torus9-ownership-ab/torus9-wdl-ownership-ab-20260913-v1/final-report.json`;
the compact committed summary is
`docs/TORUS9_OWNERSHIP_AB_20260913.json`.
