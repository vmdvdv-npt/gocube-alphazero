# TORUS9 WATCHDOG + KOMI CALIBRATION

## ARENA WATCHDOG

- legacy universal limit: **500** plies
- new board-size-scaled Torus9 limit: **1000** plies
- verification: deterministic resolver; 5×5 = 500, Torus5 = 500, Torus9 = 1000; technical limit is fail-closed and never W/L/D

## GOLDEN HISTORICAL KOMI

**0.5**. All existing trajectories were played at K=0.5. Other values below are offline counterfactual rescoring; no model was retrained.

## ESTIMATES

| Dataset | Estimated fair komi | Crossing interval | Bootstrap crossing CI | Games |
|---|---:|---:|---:|---:|
| all_512_selfplay | 0.8806 | [0.5, 1.5] | [0.72857143, 1.01968504] | 512 |
| without_D3 | 0.7529 | [0.5, 1.5] | [0.54705882, 0.96325417] | 448 |
| late_D6_D8 | 0.5556 | [0.5, 1.5] | [0.5, 0.91704545] | 192 |
| M8_related_terminal_games | — | — | [0.5, 0.7367381] | 462 |
| M8_vs_M8_paired | — | — | [0.5, 0.75738636] | 128 |

## REQUIRED READOUT

ALL SELF-PLAY ESTIMATE: mean-implied K=3.0547, median-implied K=1.0000, crossing=0.8806.
WITHOUT-D3 ESTIMATE: crossing=0.7529.
LATE-GENERATION ESTIMATE: D6-D8 crossing=0.5556; late-model agreement is interpreted with its CI, not as a forced single number.
M8-ONLY ESTIMATE: M8-related terminal trajectories crossing=—.
M8-vs-M8 PAIRED ESTIMATE: crossing=—; bootstrap unit is the independent start pair.
RECOMMENDED FAIR-KOMI RANGE: [0.5, 1.5].
BEST POINT ESTIMATE: 0.8806.
CONFIDENCE: INCONCLUSIVE.
PAIRED CONFIRMATION: NO; The 128-game paired result contradicts the 512 self-play crossing: BLACK is below 50% even at K=0.5, so no confirmation of the old range is claimed. This is INCONCLUSIVE and requires a separately designed follow-up before any more compute.

## RAW SCORE DISTRIBUTION

| Dataset | mean | median | mode | P25 | P50 | P75 |
|---|---:|---:|---|---:|---:|---:|
| all_512_selfplay | 3.0547 | 1.0000 | [1.0] | -1.0000 | 1.0000 | 3.0000 |
| without_D3 | 3.0960 | 1.0000 | [1.0] | -2.0000 | 1.0000 | 3.0000 |
| late_D6_D8 | 0.1927 | 1.0000 | [1.0] | -5.0000 | 1.0000 | 3.0000 |
| M8_related_terminal_games | 0.5195 | 0.0000 | [1.0] | -4.0000 | 0.0000 | 3.0000 |
| M8_vs_M8_paired | -0.9531 | -1.0000 | [-3.0] | -3.0000 | -1.0000 | 1.0000 |

### Exact all-512 histogram

The JSON artifact contains the same histogram as machine-readable data; the exact counts are repeated here to make the primary crossing auditable:

```json
{"-1.0": 28, "-10.0": 3, "-11.0": 2, "-12.0": 1, "-13.0": 5, "-14.0": 2, "-15.0": 1, "-16.0": 1, "-17.0": 1, "-19.0": 2, "-2.0": 11, "-20.0": 1, "-22.0": 1, "-23.0": 1, "-26.0": 1, "-27.0": 4, "-3.0": 16, "-37.0": 1, "-4.0": 11, "-42.0": 1, "-43.0": 1, "-44.0": 2, "-45.0": 1, "-47.0": 1, "-5.0": 16, "-6.0": 4, "-7.0": 10, "-8.0": 6, "-9.0": 6, "0.0": 64, "1.0": 134, "10.0": 4, "11.0": 1, "12.0": 3, "13.0": 3, "14.0": 2, "15.0": 3, "16.0": 2, "18.0": 1, "2.0": 29, "20.0": 1, "21.0": 1, "23.0": 1, "24.0": 1, "25.0": 3, "26.0": 1, "29.0": 1, "3.0": 32, "30.0": 2, "31.0": 1, "33.0": 2, "34.0": 2, "35.0": 2, "36.0": 1, "38.0": 1, "39.0": 3, "4.0": 13, "41.0": 3, "42.0": 1, "43.0": 1, "45.0": 1, "48.0": 2, "5.0": 13, "51.0": 1, "53.0": 1, "6.0": 4, "7.0": 9, "8.0": 9, "81.0": 9, "9.0": 3}
```

## GENERATIONS D1–D8

| Generation | BLACK rate at K=0.5 | mean raw advantage | median raw advantage | crossing interval |
|---|---:|---:|---:|---|
| D1 | 0.5000 | 5.3906 | 0.5000 | [0.5, 0.5] |
| D2 | 0.6562 | 0.9688 | 1.0000 | [0.5, 1.5] |
| D3 | 0.9531 | 2.7656 | 1.0000 | [0.5, 1.5] |
| D4 | 0.4531 | 6.1094 | 0.0000 | — |
| D5 | 0.7031 | 8.6250 | 3.0000 | [2.5, 3.5] |
| D6 | 0.4531 | -2.0469 | -1.0000 | — |
| D7 | 0.4219 | -1.7344 | -1.0000 | — |
| D8 | 0.6562 | 4.3594 | 1.0000 | [0.5, 1.5] |

## WINNER FLIPS

At the all-512 best point, **0 / 512** existing self-play trajectories change WDL under offline rescoring from K=0.5. This does not rewrite training targets.
At the upper observed crossing boundary K=1.5, **134 / 512** self-play trajectories flip relative to K=0.5; this is why the median and histogram matter more than mean raw margin.

## SCIENTIFIC LIMITS

The komi estimate is trajectory-conditioned: a model trained and searched with another real komi could choose other moves. Existing technical Arena games are excluded from W/L/D; if their terminal continuation is used, it remains sensitivity analysis, not an Arena played under that komi.

Would this justify changing training komi in the next separate scientific run? **INCONCLUSIVE**. The calibration supports a diagnostic range, but a future change needs a separately specified training experiment and must not reinterpret the frozen M8 Golden result.

Source commit: `d1fd6210f6a3b7755341c0c4c832ed22cf5289d4`; source tree: `a8e193a0bef2ba2b52f562eae528e5b7dfc4f33b`; report schema: `torus9-komi-calibration-v1`.
