# Torus 9×9 Golden learning proof

TORUS 9×9 CONTRACT: PASS

POINTS: 81
ACTIONS: 82
PASS INDEX: 81
KOMI: 0.5

BRING-UP: PASS

BRING-UP:
5 × 32 games

MEAN POSITIONS/GAME: 75.48125
PROJECTED POSITIONS @ 64: 4830.8
SELECTED CANONICAL GAMES/ITER: 64
WHY: 64 games is the frozen canonical setting and projects to approximately 4.8k fresh positions, near the lower edge of the requested range.

| ITER | GAMES | POSITIONS | POS/GAME | SELF-PLAY TIME | TRAIN TIME | TECHNICAL |
|---:|---:|---:|---:|---:|---:|---:|
| M1 | 32 | 1929 | 60.28 | 45.49s | 0.73s | 0 |
| M2 | 32 | 3555 | 111.09 | 76.85s | 1.36s | 0 |
| M3 | 32 | 2342 | 73.19 | 58.11s | 0.87s | 0 |
| M4 | 32 | 2295 | 71.72 | 54.36s | 0.88s | 0 |
| M5 | 32 | 1956 | 61.12 | 44.62s | 0.74s | 0 |

## Canonical M0→M8

| ITER | GAMES | POSITIONS | OPT STEPS | AVG PLY | SELF-PLAY TIME | TRAIN TIME | TECHNICAL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 5265 | 83 | 82.27 | 117.45s | 2.03s | 0 |
| 2 | 64 | 7348 | 115 | 114.81 | 161.58s | 2.95s | 0 |
| 3 | 64 | 1962 | 31 | 30.66 | 47.24s | 0.77s | 0 |
| 4 | 64 | 2763 | 44 | 43.17 | 67.63s | 1.11s | 0 |
| 5 | 64 | 5033 | 79 | 78.64 | 116.05s | 1.94s | 0 |
| 6 | 64 | 7470 | 117 | 116.72 | 158.16s | 3.02s | 0 |
| 7 | 64 | 3727 | 59 | 58.23 | 87.80s | 1.53s | 0 |
| 8 | 64 | 7342 | 115 | 114.72 | 151.24s | 3.01s | 0 |

CANONICAL RUN: M0 → M8
GAMES/ITER: 64
TOTAL SELF-PLAY: 512
TOTAL POSITIONS: 40910
TECHNICAL: 0 expected

## Arena report

M4 vs M0 = [6, 58, 0] | pairs=32 | mean=0.09375 | CI=[0.0, 0.3338306978299802] | technical=0
M4 vs M1 = [8, 24, 0] | pairs=16 | mean=0.25 | CI=[0.0, 0.5895253789351549] | technical=0
M8 vs M0 = [75, 53, 0] | pairs=64 | mean=0.5859375 | CI=[0.41617481053242256, 0.7557001894675774] | technical=0
M8 vs M1 = [62, 57, 0] | pairs=55 | mean=0.5181818181818182 | CI=None | technical=9
M8 vs M4 = [80, 48, 0] | pairs=64 | mean=0.625 | CI=[0.45523731053242256, 0.7947626894675774] | technical=0
M8 vs M7 = [10, 9, 0] | pairs=6 | mean=0.5833333333333334 | CI=None | technical=13

PRIMARY CONFIDENCE INTERVALS:
- M8-vs-M0: [0.41617481053242256, 0.7557001894675774]
- M8-vs-M1: None
- M8-vs-M4: [0.45523731053242256, 0.7947626894675774]

## Verdict

TORUS 9×9 LEARNING: NOT CONFIRMED

WHY: M8-vs-M0 primary=False; M4 movement=False; M8-vs-M4 non-regression=True; technical self-play and replay validation passed.

## First-move report

TORUS 9×9 FIRST-MOVE OBSERVATION

KOMI: 0.5
BLACK WIN RATE: 0.46484375
95% CI: [0.4220636212637802, 0.508147494216735]
RAW BLACK AREA ADVANTAGE: 0.728515625
FINAL BLACK MARGIN: 0.228515625
ADVANTAGE: INCONCLUSIVE

## Performance

games/hour: 2407.6375976523514
positions/hour: 159427.104694602
avg ply: 79.90234375
Arena workers: 16
Peak memory MB: 1683.08203125

## Representative game traces

- `selfplay-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-01.json`
- `selfplay-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-02.json`
- `selfplay-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-03.json`
- `selfplay-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-04.json`
- `selfplay-05.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-05.json`
- `selfplay-06.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/selfplay-06.json`
- `M4-vs-M0-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M0-game-01.json`
- `M4-vs-M0-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M0-game-02.json`
- `M4-vs-M0-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M0-game-03.json`
- `M4-vs-M0-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M0-game-04.json`
- `M4-vs-M1-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M1-game-01.json`
- `M4-vs-M1-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M1-game-02.json`
- `M4-vs-M1-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M1-game-03.json`
- `M4-vs-M1-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M4-vs-M1-game-04.json`
- `M8-vs-M0-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M0-game-01.json`
- `M8-vs-M0-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M0-game-02.json`
- `M8-vs-M0-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M0-game-03.json`
- `M8-vs-M0-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M0-game-04.json`
- `M8-vs-M1-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M1-game-01.json`
- `M8-vs-M1-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M1-game-02.json`
- `M8-vs-M1-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M1-game-03.json`
- `M8-vs-M1-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M1-game-04.json`
- `M8-vs-M4-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M4-game-01.json`
- `M8-vs-M4-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M4-game-02.json`
- `M8-vs-M4-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M4-game-03.json`
- `M8-vs-M4-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M4-game-04.json`
- `M8-vs-M7-game-01.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M7-game-01.json`
- `M8-vs-M7-game-02.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M7-game-02.json`
- `M8-vs-M7-game-03.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M7-game-03.json`
- `M8-vs-M7-game-04.json`: `/home/codex/projects/gocube-alphazero/runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3/canonical/representative-traces/M8-vs-M7-game-04.json`

## Verification

TARGETED TESTS: 5 passed in 1.44s
FULL PYTEST: 1217 passed, 1 failed, 57 warnings in 359.99s
FULL PYTEST NOTE: the sole failure is the pre-existing data-dependent fork audit (`8208` persisted fork records versus an assertion of zero); no Torus 9×9 test failed.
SOURCE SHA: 9723bb5ac8eb28d55d21d607e3673da5bd894315
FINAL COMMIT: 9723bb5ac8eb28d55d21d607e3673da5bd894315
PR: pending final push
CI: pending final push
