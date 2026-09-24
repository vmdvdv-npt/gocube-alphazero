# Torus9 Black/White search-source diagnostic

> Read-only diagnostic on 256 saved M135 self-play positions. It does not modify training artifacts, run games, or alter production configuration.

## Direct result

The four-stage comparison below is the primary evidence. Deltas are `WHITE - BLACK`; compact aggregate delta summaries are retained in JSON.

- Raw NN policy: see `raw_nn_policy` aggregate metrics; raw WDL/value: see `raw_wdl_value`.
- Dirichlet: see `noisy_prior` aggregate metrics across the three repeats.
- MCTS with noise ON/OFF: see `mcts_noise_on` and `mcts_noise_off`.
- Checkpoint: `M134` `sha256:27dc291ff99da846fb554aad7ab7c071d8177424e58489bb5b45938112984625`.

## Lineage and effective search

- Run: `torus9-m125-continuous-v2-gen6-20260922-v1`; self-play generation: `M135`.
- Checkpoint path: `/home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-m125-continuous-v2-gen6-20260922-v1/checkpoints/M134.pt`.
- Git commit recorded by self-play: `706dc5c38b6846f4d0ac3507d7f0c50a55291310`; tree: `32752fe1b3d14fc2b394a8198ef93e4b0a7f3754`.
- Search: Torus9, komi 0.5, 200 simulations, cpuct 1.25, FPU 0, Dirichlet α/ε 0.11/0.25, action space 82.

## Aggregate stage comparison

The compact table reports mean entropy, effective candidates, top-1 share and top-3 share. Aggregate quantiles and Black/White deltas are in the JSON; raw vectors are intentionally omitted.

| Stage | BLACK H | WHITE H | ΔH | BLACK eff. | WHITE eff. | Δeff. | BLACK top-1 | WHITE top-1 | Δtop-1 | BLACK top-3 | WHITE top-3 | Δtop-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_nn_policy | 2.438 | 3.146 | +0.708 | 17.78 | 34.53 | +16.75 | 0.353 | 0.216 | -0.137 | 0.568 | 0.367 | -0.201 |
| noisy_prior | 2.712 | 3.193 | +0.481 | 19.96 | 32.86 | +12.89 | 0.288 | 0.196 | -0.092 | 0.498 | 0.368 | -0.130 |
| mcts_noise_on | 1.813 | 2.821 | +1.008 | 8.51 | 26.39 | +17.87 | 0.446 | 0.287 | -0.159 | 0.691 | 0.465 | -0.226 |
| mcts_noise_off | 1.553 | 2.801 | +1.248 | 6.94 | 28.90 | +21.96 | 0.505 | 0.292 | -0.212 | 0.749 | 0.452 | -0.297 |

## By ply range

| Range | Stage | BLACK H | WHITE H | ΔH | BLACK eff. | WHITE eff. | Δeff. |
|---|---|---:|---:|---:|---:|---:|---:|
| plies_1_8 | raw_nn_policy | 2.952 | 4.017 | +1.064 | 33.93 | 67.05 | +33.12 |
| plies_1_8 | noisy_prior | 3.250 | 3.974 | +0.725 | 34.97 | 59.47 | +24.51 |
| plies_1_8 | mcts_noise_on | 2.162 | 3.787 | +1.625 | 11.59 | 50.84 | +39.24 |
| plies_1_8 | mcts_noise_off | 1.655 | 3.941 | +2.286 | 6.84 | 62.47 | +55.63 |
| plies_9_24 | raw_nn_policy | 2.379 | 3.558 | +1.179 | 14.40 | 39.69 | +25.29 |
| plies_9_24 | noisy_prior | 2.816 | 3.614 | +0.798 | 19.42 | 39.63 | +20.21 |
| plies_9_24 | mcts_noise_on | 1.487 | 3.316 | +1.828 | 5.09 | 31.94 | +26.86 |
| plies_9_24 | mcts_noise_off | 1.193 | 3.217 | +2.023 | 3.74 | 30.40 | +26.66 |
| plies_25_64 | raw_nn_policy | 2.288 | 2.780 | +0.491 | 11.18 | 19.70 | +8.52 |
| plies_25_64 | noisy_prior | 2.589 | 2.924 | +0.336 | 14.21 | 21.08 | +6.87 |
| plies_25_64 | mcts_noise_on | 1.648 | 2.310 | +0.662 | 7.30 | 13.54 | +6.24 |
| plies_25_64 | mcts_noise_off | 1.481 | 2.196 | +0.715 | 6.53 | 12.77 | +6.24 |
| plies_65_plus | raw_nn_policy | 2.132 | 2.230 | +0.098 | 11.61 | 11.69 | +0.08 |
| plies_65_plus | noisy_prior | 2.193 | 2.260 | +0.066 | 11.27 | 11.24 | -0.03 |
| plies_65_plus | mcts_noise_on | 1.954 | 1.872 | -0.082 | 10.08 | 9.23 | -0.85 |
| plies_65_plus | mcts_noise_off | 1.882 | 1.849 | -0.033 | 10.64 | 9.96 | -0.69 |

## Interpretation guardrails

- `komi=0.5` creates a real Black/White asymmetry; a color gap is not automatically an implementation bug.
- The diagnostic localizes mechanisms. It does not establish causality for any training-speed plateau.
- No production parameter, training artifact, checkpoint, self-play record, or Arena result was changed or created.

## Answers to the requested questions

1. **Raw NN policy:** yes. Aggregate Black→White entropy is `+0.708` nats and effective candidates `+16.75`; the gap is already large before noise or MCTS.
2. **Raw WDL/value:** yes, as a statistical distributional difference. The aggregate median derived utility is `+0.147` for BLACK-to-move versus `-0.150` for WHITE-to-move. These are different positions, so this is not an assertion that paired game situations are equivalent.
3. **Dirichlet:** it does not create the gap. It reduces the aggregate entropy delta from `+0.708` to `+0.481` nats and the effective-candidate delta from `+16.75` to `+12.89`; repeat distributions are in `noisy_prior`.
4. **Noise OFF:** the gap remains and is larger: entropy delta `+1.248` nats, versus `+1.008` with noise ON.
5. **Sharp increase:** the main increase occurs in PUCT/MCTS. Aggregate entropy deltas are raw `+0.708`, noisy prior `+0.481`, MCTS ON `+1.008`, MCTS OFF `+1.248`; the strongest effects are in plies 1–24.
6. **P versus Q/value:** final visits are more directly aligned with prior P than with Q at this budget. For MCTS noise ON, mean per-position Pearson P→visits is about `0.90` for both colors, while Q→visits is `0.39` BLACK and `0.24` WHITE. Q/value still participates: WHITE has the wider root-Q spread, so this is a P-dominant PUCT interaction rather than a claim that Q is irrelevant.
7. **Implementation bug evidence:** none was found. The diagnostic had zero raw-shape errors, illegal root visits, or root-visit-sum errors; existing state-chain/legal/PASS checks also remain clean.
8. **Most likely source:** a combination dominated by learned network policy, with raw value/Q contributing inside PUCT. Dirichlet is an amplifier/variance source here, not the primary source.
9. **Production change:** no basis for changing production parameters from this localization alone. `komi=0.5` is a real asymmetry source and this experiment does not establish undesired behavior.
10. **Training slowdown:** no new evidence of a causal connection to M130+ training slowdown. This diagnostic localizes search asymmetry only.

The compact JSON contains lineage/config, sample balance, aggregate and ply-range metrics, Black/White deltas, Q/correlation summaries, validation invariants, and conclusions. Raw vectors, legal masks, per-position rows, and full paired distributions are intentionally not tracked.
