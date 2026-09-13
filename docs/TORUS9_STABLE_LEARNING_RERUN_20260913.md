TORUS 9×9 STABLE LEARNING RERUN

SOURCE:
25dc804cced364c4289e4b0580e3799c4437e288 (clean source commit)

FIXES:
8 graph blocks
3-generation / 20k rolling replay
80 full batch64 updates per iteration

SELF-PLAY:
8 × 64 = 512 games

OVERALL: LEARNING CONFIRMED

TRAINING STABILITY: PARTIAL

OLD M8 vs NEW M8: [59, 3, 0]

OLD FAILURE MODE REPRODUCED: NO
M4-LIKE COLLAPSE: NO

## Contract and run

Architecture: GoldenGraphNetV2-Torus9-8Block / sha256:0d744308071dcdafc574159c09d0112f61fcf2823bd7d43aba8da4bd50eaddac
Policy/value heads: [82] / [3]; PASS index: 81; komi: 0.5
Replay: 3 generations, cap 20000 positions
Optimizer: Adam lr=0.001 wd=0; 80 × batch 64

## M0→M8 telemetry

| ITER | GAMES | FRESH POS | REPLAY POS | AVG PLY | WINNERS | OPT STEPS | SAMPLES | UNIQUE / REUSED | WDL BIAS |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| M1 | 64 | 5912 | 5912 | 92.38 | {'BLACK': 32, 'WHITE': 32} | 80 | 5120 | 5120 / 0 | 0.1354 |
| M2 | 64 | 1554 | 7466 | 24.28 | {'BLACK': 42, 'WHITE': 22} | 80 | 5120 | 5120 / 0 | 0.0416 |
| M3 | 64 | 3407 | 10873 | 53.23 | {'BLACK': 61, 'WHITE': 3} | 80 | 5120 | 5120 / 0 | 0.4693 |
| M4 | 64 | 4967 | 9928 | 77.61 | {'BLACK': 29, 'WHITE': 35} | 80 | 5120 | 5120 / 0 | 0.3095 |
| M5 | 64 | 6695 | 15069 | 104.61 | {'BLACK': 45, 'WHITE': 19} | 80 | 5120 | 5120 / 0 | 0.5675 |
| M6 | 64 | 6957 | 18619 | 108.70 | {'BLACK': 29, 'WHITE': 35} | 80 | 5120 | 5120 / 0 | 0.0320 |
| M7 | 64 | 8103 | 20000 | 126.61 | {'BLACK': 27, 'WHITE': 37} | 80 | 5120 | 5120 / 0 | 0.0233 |
| M8 | 64 | 5239 | 20000 | 81.86 | {'BLACK': 42, 'WHITE': 22} | 80 | 5120 | 5120 / 0 | -0.0320 |

## Arena

| COMPARISON | W/L/D | VALID PAIRS | TECHNICAL GAMES | MEAN PAIR SCORE | 95% INTERVAL |
|---|---:|---:|---:|---:|---|
| M4-vs-M0 | [46, 18, 0] | 32 | 0 | 0.71875 | [0.4786693021700198, 0.9588306978299802] |
| M4-vs-M1 | [32, 0, 0] | 16 | 0 | 1.0 | [0.6604746210648451, 1.0] |
| M8-vs-M0 | [119, 7, 0] | 62 | 2 | 0.9435483870967742 | None |
| M8-vs-M1 | [128, 0, 0] | 64 | 0 | 1.0 | [0.8302373105324226, 1.0] |
| M8-vs-M4 | [92, 24, 0] | 52 | 12 | 0.8269230769230769 | None |
| M8-vs-M7 | [23, 7, 0] | 14 | 2 | 0.7857142857142857 | None |
| NEW-M8-vs-OLD-M8 | [59, 3, 0] | 30 | 2 | 0.95 | None |

## Old/new trajectory

| ITER | OLD POS | NEW POS | OLD AVG PLY | NEW AVG PLY | OLD WDL BIAS | NEW WDL BIAS |
|---:|---:|---:|---:|---:|---:|---:|
| M1 | 5265 | 5912 | 82.265625 | 92.38 | -0.2042198828421533 | 0.1354 |
| M2 | 7348 | 1554 | 114.8125 | 24.28 | 0.033202324993908405 | 0.0416 |
| M3 | 1962 | 3407 | 30.65625 | 53.23 | 0.3261291701346636 | 0.4693 |
| M4 | 2763 | 4967 | 43.171875 | 77.61 | 0.6648688621353358 | 0.3095 |
| M5 | 5033 | 6695 | 78.640625 | 104.61 | -0.3381112557835877 | 0.5675 |
| M6 | 7470 | 6957 | 116.71875 | 108.70 | 0.017263561952859163 | 0.0320 |
| M7 | 3727 | 8103 | 58.234375 | 126.61 | -0.7680078900884837 | 0.0233 |
| M8 | 7342 | 5239 | 114.71875 | 81.86 | -0.5725909136235714 | -0.0320 |

## Final analysis

1. 8-block network global point-policy coverage: CONFIRMED by the Torus5 diameter=4 / Torus9 diameter=8 dependency proof and canonical 8-block path.
2. Fresh-generation feedback collapse: PARTIAL; inspect the fixed-state and replay telemetry.
3. Training-impulse wandering: NOT REPRODUCED; optimizer exposure is fixed at 80×64.
4. WDL oscillation: NO under the declared threshold.
5. NEW M8 stronger than OLD M8: 0.95 mean paired score.
6. NEW M8 stronger than NEW M1: 1.0.
7. NEW M4 normal or degraded: NORMAL.
8. Pipeline ready for next stage: NOT YET; retain the diagnostic boundary.
9. Remaining bottleneck: 64-simulation teacher/search target quality remains the known unmodified risk.
10. Next step: better teacher/search diagnostics first; ownership, history, and other auxiliary changes remain out of this rerun.

## Verification

Cheap contract: PASS
Performance preflight: PASS
Source commit: 25dc804cced364c4289e4b0580e3799c4437e288
Targeted Torus/forensic pytest: 12 passed in 9.50s
Full pytest: 1 failed, 1224 passed, 57 warnings in 353.44s; the sole failure is the pre-existing ignored-data fixture expecting 0 fork records while `data/` contains 8208 current fork records. No task code or ignored data was changed.
