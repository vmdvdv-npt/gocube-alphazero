# Golden Torus Stage 3 diagnostic

This is a retrospective diagnosis of the immutable canonical run
`runs/torus-golden-stage3/torus-golden-stage3-seed1-v5/`. It does not rewrite
the historical Stage 3 verdict (`LEARNING NOT DEMONSTRATED`) and does not
modify any Stage 3 checkpoint, replay, Arena record, or original report.

## Confirmed methodology defects

The original Arena used nested deterministic prefixes of one trajectory. Its
starts were therefore strongly dependent and did not constitute an
independent/diverse corpus. The old Arena also reported
`mean ± 1.96 * sample_sd / sqrt(n)` over eight pair scores. With all pair
scores equal to `0.5`, that formula produced `[0.5, 0.5]`; it was not a valid
population-strength confidence statement for this dependent, deterministic
sample.

The original `M4 vs M1` result was 8–8, with all 16 games won by Black. This
is evidence of color asymmetry and low color-controlled discriminative power
of that small start set. It is not evidence that Black intrinsically wins
Torus 5×5, that komi is wrong, or that the rules are wrong.

## Training diagnosis

Stage 3 generated 64 valid self-play games and 2,055 replay positions. It
consumed 800 × 128 = 102,400 sampled examples, approximately 49.83 sampled
examples per generated position. Chunk 1 alone consumed 25,600 samples for
497 new positions (approximately 51.51 samples per new position). The
replay policy and training losses are useful consistency diagnostics, but are
not independent evidence of playing strength.

The historical replay metrics show most policy cross-entropy reduction by M1:

| checkpoint | policy CE | value CE | value accuracy |
|---|---:|---:|---:|
| M0 | 3.230 | 1.138 | 0.300 |
| M1 | 2.217 | 0.485 | 0.735 |
| M2 | 2.196 | 0.438 | 0.775 |
| M3 | 2.186 | 0.416 | 0.787 |
| M4 | 2.174 | 0.376 | 0.811 |

This is consistent with an early replay-fitting effect, but it cannot by
itself establish a plateau or its cause.

## Stage 4 correction

The controlled continuation uses the versioned
`golden-evaluation-startset-v2` contract and
`gocube-torus-golden-training-v2-data-rich` profile. Evaluation V2 uses 64
independent legal non-pass prefix trajectories, stratified as eight starts at
each length 2, 4, 6, 8, 10, 12, 14, and 16. Exact deduplication includes the
complete Golden state identity, including full positional-superko history.
The 200 full-history Torus automorphisms are implemented and tested as a
diagnostic. They are not used to reject candidates because prefix length 2
cannot supply eight distinct symmetry classes; using them as a rejection rule
would make the requested corpus impossible.

The primary pair statistic is the mean over color-swapped start pairs. The
primary uncertainty report is the conservative 95% Hoeffding bounded-mean
interval; the old zero-variance normal interval is not used.

The new run separately tests more unique fresh data (A vs B), high reuse vs
one-epoch low reuse (B vs C), and independent-seed Stage 4 progression. The
Golden replay audit is an artifact-consistency audit against the same Golden
rules implementation, not an independent rules oracle.
