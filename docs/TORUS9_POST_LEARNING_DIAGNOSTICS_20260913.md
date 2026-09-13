TORUS 9×9 POST-LEARNING DIAGNOSTICS

CURRENT GOLDEN BEST:
torus9-stable-learning-20260913-v1 / M8
sha256:0c2beca91354b04b26f7092c94b5c60b217c5db2d5e8cb5403a124bb39738761

SELF-PLAY 61-3 ROOT CAUSE:
A + C with contributing D: expected first-player/komi advantage is amplified by the small 64-game sample and early stochastic PASS branches. Confidence: STRONG EVIDENCE. It is not a two-copy M2 mismatch, rules issue, or single-opening collapse.

TRUNCATION ROOT CAUSE:
B + C + D: legal long-play/capture churn and single-PASS continuation, with pair-specific PASS avoidance. Confidence: STRONG EVIDENCE. Superko activity is not the primary loop mechanism.

64-SIM TEACHER VERDICT:
64-sim teacher remains a live quality risk: fixed-state 64/128/256 differences are reported, especially D3/M2 and truncation-tail states. No baseline change is authorized by this report alone.

FIRST-PLAYER ADVANTAGE:
Across all 512 self-play games BLACK win rate is 0.5996, Wilson 95% CI [0.55657805, 0.64115713]; excluding D3 it is 0.5491 with CI [0.50280808, 0.59457121]. Average raw area advantage is 3.0546875 and average final komi-adjusted margin is 2.5546875. D3 is an extreme realization, not evidence of a different M2 on the two colors.

NEXT RECOMMENDED CHANGE:
One isolated better-teacher/search experiment (64 → 128 simulations) after this diagnostic is reviewed; keep architecture, replay, optimizer, komi, noise, temperature and canonical 500-ply protocol unchanged in the comparison baseline.

CONFIDENCE:
STRONG EVIDENCE for the causal classifications; CONFIRMED for provenance/rules/technical separation; PLAUSIBLE for the exact share attributable to network calibration versus search/noise; UNKNOWN for whether 128-sim targets win a future canonical comparison.

## CONFIRMED

- PR #86 is represented by merge anchor `88b1803cf179c6fa93f2e9610963eeed931d09b1`; the artifact-producing source commit is recorded separately.
- M2/D3 used one model hash on both sides: ['sha256:e3d6528630b417b056b54140e282b23f4397dab50e4d099738c5cfb3eba0299d'].
- D3 is 61/3/0, with the three WHITE results all early PASS branches; no dominant first action explains the result.
- M2 empty-board network root WDL is [0.53289849, 0.00176452, 0.46533701] (utility 0.06756148, PASS probability 0.02282308); the early value bias is finite, not a deterministic forced-win signal.
- Across all 512 self-play games, BLACK is 0.5996 with Wilson 95% CI [0.55657805, 0.64115713]; raw area advantage 3.0546875, final komi-adjusted margin 2.5546875.
- Excluding D3, BLACK is 0.5491 (Wilson 95% CI [0.50280808, 0.59457121]); generation-index correlation is -0.0070, so checkpoint strength does not explain a monotonic color drift.
- Canonical Arena technical games remain excluded from W/L/D.

## STRONG EVIDENCE

### Self-play color balance

| checkpoint | generation | BLACK | WHITE | draw | BLACK rate | Wilson 95% CI | avg margin | median margin | avg plies | pass frequency |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| M0 | D1 | 32 | 32 | 0 | 0.5000 | [0.38102095, 0.61897905] | 4.891 | 0.000 | 92.38 | 0.1003 |
| M1 | D2 | 42 | 22 | 0 | 0.6562 | [0.5340364, 0.76076858] | 0.469 | 0.500 | 24.28 | 0.1036 |
| M2 | D3 | 61 | 3 | 0 | 0.9531 | [0.87100346, 0.98393098] | 2.266 | 0.500 | 53.23 | 0.0754 |
| M3 | D4 | 29 | 35 | 0 | 0.4531 | [0.33729448, 0.57426403] | 5.609 | -0.500 | 77.61 | 0.0763 |
| M4 | D5 | 45 | 19 | 0 | 0.7031 | [0.58229799, 0.80094849] | 8.125 | 2.500 | 104.61 | 0.0842 |
| M5 | D6 | 29 | 35 | 0 | 0.4531 | [0.33729448, 0.57426403] | -2.547 | -1.500 | 108.70 | 0.0887 |
| M6 | D7 | 27 | 37 | 0 | 0.4219 | [0.30869782, 0.54389968] | -2.234 | -1.500 | 126.61 | 0.1423 |
| M7 | D8 | 42 | 22 | 0 | 0.6562 | [0.5340364, 0.76076858] | 3.859 | 0.500 | 81.86 | 0.0977 |

D3 game-level forensic data (including seeds, first eight actions, early root WDL, visit share and PASS data) is in the JSON report; the short WHITE branches are the only D3 games with first PASS at ply ≤5.

### Fixed-state 64/128/256 search

| source | comparison | states | mean policy KL | top-1 agreement | mean top-3 overlap | mean |Δ root value| | mean |Δ PASS| |
|---|---|---:|---:|---:|---:|---:|---:|
| ARENA_TRUNCATED | 128_vs_256 | 32 | 0.17838 | 0.844 | 0.854 | 0.01629 | 0.07361 |
| ARENA_TRUNCATED | 64_vs_128 | 32 | 0.09715 | 0.750 | 0.885 | 0.01473 | 0.05029 |
| ARENA_TRUNCATED | 64_vs_256 | 32 | 0.27043 | 0.719 | 0.802 | 0.02644 | 0.09387 |
| D3 | 128_vs_256 | 8 | 0.03496 | 0.875 | 0.708 | 0.02053 | 0.00195 |
| D3 | 64_vs_128 | 8 | 0.06048 | 0.875 | 0.667 | 0.00909 | 0.00195 |
| D3 | 64_vs_256 | 8 | 0.02689 | 1.000 | 0.917 | 0.02962 | 0.00391 |
| D4 | 128_vs_256 | 8 | 0.43971 | 1.000 | 1.000 | 0.03975 | 0.00391 |
| D4 | 64_vs_128 | 8 | 0.46976 | 0.500 | 0.250 | 0.04478 | 0.00781 |
| D4 | 64_vs_256 | 8 | 0.70967 | 0.500 | 0.250 | 0.08453 | 0.01172 |
| D6 | 128_vs_256 | 8 | 0.16781 | 0.375 | 0.500 | 0.00874 | 0.00391 |
| D6 | 64_vs_128 | 8 | 0.22357 | 0.000 | 0.125 | 0.01635 | 0.00781 |
| D6 | 64_vs_256 | 8 | 0.05834 | 0.125 | 0.250 | 0.02335 | 0.01172 |
| D7 | 128_vs_256 | 8 | 0.01715 | 0.125 | 0.417 | 0.00946 | 0.00391 |
| D7 | 64_vs_128 | 8 | 0.05552 | 0.500 | 0.583 | 0.02473 | 0.00781 |
| D7 | 64_vs_256 | 8 | 0.05868 | 0.125 | 0.292 | 0.01527 | 0.00781 |

### Noise / temperature controlled diagnostic on fixed M2 states

| variant | states | repeats/state | mean unique actions | mean decision entropy | PASS decision rate | mean search PASS probability |
|---|---:|---:|---:|---:|---:|---:|
| A_64_noise_off_temperature_0 | 16 | 16 | 1.000 | 0.00000 | 0.12500 | 0.13867 |
| B_64_noise_on_temperature_0 | 16 | 16 | 12.688 | 2.29942 | 0.12500 | 0.13873 |
| C_64_noise_off_temperature_1 | 16 | 16 | 12.750 | 2.30688 | 0.14062 | 0.13867 |
| D_64_noise_on_temperature_1 | 16 | 16 | 12.875 | 2.31362 | 0.14062 | 0.13873 |

Interpretation: A is deterministic; adding either root noise (B) or temperature sampling (C) raises decision entropy to about 2.3, while PASS decision rates remain close to the fixed-state search PASS probability. The measured instability is therefore branch selection, not a standalone PASS-probability explosion.

### Arena truncations

| classification | games | representative evidence |
|---|---:|---|
| CAPTURE_CYCLE | 4 | See per-game last-100 metrics and search samples in JSON. |
| PASS_AVOIDANCE | 14 | See per-game last-100 metrics and search samples in JSON. |

All technical games reach ply 500 through legal nonterminal transitions. Exact board signatures repeat only on PASS-retained boards; point-move board repetitions are absent, and superko rejections are activity but not a loop mechanism.

### Offline continuation

18 games finished by ply 1000; 0 after ply 1000 and before 1600; 0 remained technical at 1600. Canonical Arena evidence was not modified.

## PLAUSIBLE

- D3 is best explained by a real first-player/komi edge plus small-sample and early stochastic branch amplification. Noise and temperature measurements quantify the contribution; they do not justify changing the frozen baseline.
- Truncation is a mixture of PASS avoidance and capture/territory-filling churn, with relative proportions varying by pair; it is not one universal cycle.

## REJECTED

- Two different M2 copies with unequal strength: rejected by the single model hash and identical contract provenance on both sides.
- A single dominant opening move as the cause of 61–3: rejected by the D3 first-eight action distributions.
- Rules/scoring or superko implementation failure as the truncation cause: rejected by legal replay, no point-board repetition, and fail-closed technical handling.
- Changing komi, move limit, architecture, replay, optimizer or self-play settings during diagnosis: not done.

## UNKNOWN

- A causal claim that 128 simulations would improve future training targets is not established by these fixed-state probes alone; it is the next clean experiment if a search change is approved.
- A human strategic quality judgment is limited to the linked visual snapshots; no claim of game-theoretic optimality is made.

## Decision matrix

- Self-play 61–3: A. Expected first-player / stochastic variance, with D. network/value calibration as a plausible contributor to early PASS branch amplification.
- Arena truncations: B. Weak 64-sim search + C. PASS avoidance + D. value/policy pathology; A. move-limit too low may also apply to games that finish shortly after 500, to be judged from continuation results.

## Visual inspection

Visual trace artifact: `/home/codex/projects/gocube-alphazero/docs/assets/TORUS9_POST_LEARNING_VISUAL_TRACES.svg`; preview PNG: `/home/codex/projects/gocube-alphazero/docs/assets/TORUS9_POST_LEARNING_VISUAL_INSPECTION.png`. Scope: 2 ordinary current M8 games, 2 ordinary OLD M8 games, 3 D3 games (including typical and short branches), and 7 representative truncated Arena games.

The snapshots show normal legal placement/capture dynamics in current M8 and OLD M8 Arena games and in D3. In this selected visual sample, current M8 games terminate at plies 16–17, while the selected OLD M8 games include a 337-ply game. Truncation panels show either single-PASS continuation or high-capture late churn rather than an exact point-state loop.

## Verification and provenance

Baseline manifest: `/home/codex/projects/gocube-alphazero/docs/TORUS9_GOLDEN_BEST.json`
Analysis source commit: `0c27b2c34e9101c2beaf04b3637e3e75ef3a83f7`
Baseline artifact source commit: `25dc804cced364c4289e4b0580e3799c4437e288`
Baseline merge anchor: `88b1803cf179c6fa93f2e9610963eeed931d09b1`
Rules fingerprint: `sha256:e0fd15c82d42a63ca05c3b6fb3ae02deb938543e1e06483ecfeab275dc98a39e`; observation fingerprint: `sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f`; komi: `0.5`.
No new M0→M8 training run or new self-play games were launched.
