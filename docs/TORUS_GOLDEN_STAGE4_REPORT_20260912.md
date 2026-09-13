# Stage 3→4 Controlled Learning Diagnosis + Independent Confirmation

Run: `torus-golden-stage4-seed2-v4`  
Global verdict: **LEARNING SYSTEM CONFIRMED**  
Pipeline validity: **PASS**

## Conclusions

| Hypothesis | Verdict |
|---|---|
| H1 — Stage 3 evaluation starts were dependent | CONFIRMED |
| H2 — the old CI could be degenerate | CONFIRMED |
| H3 — Stage 3 signal was color-masked | INCONCLUSIVE |
| H4 — Stage 3 improvement was real on the corrected subset | CONFIRMED |
| H5 — high reuse on low fresh-data volume caused the plateau | REFUTED |
| H6 — unique-data volume matters at fixed optimizer budget | CONFIRMED |
| H7 — low reuse alone is the complete remedy | INCONCLUSIVE |
| H8 — learning replicates under an independent seed | REPLICATED |
| H9 — the new lineage improves incrementally | DEMONSTRATED |

The Golden neural pipeline technically works and learns meaningful strength from random initialization. The Stage 3 M1→M4 plateau was real, but this run does not support attributing it to high optimizer reuse alone. The data-rich profile is the recommended baseline for subsequent stages.

## Evaluation V2

- Frozen independent legal non-pass prefixes: **64**, with 8 starts at each length 2/4/6/8/10/12/14/16.
- Exact deduplication uses the complete Golden state identity, including full positional-superko history. One candidate was rejected as an exact semantic duplicate.
- The 200 Torus automorphisms (translations × D4) transform the current board and every history board. Symmetry is diagnostic-only because prefix length 2 has fewer than eight possible classes; it is not used to reject starts.
- Pair Arena uses color swaps, independent frozen start-pairs, and the primary bounded-mean 95% Hoeffding interval. The old zero-variance CI is never used.
- Empty-board control is reported separately and is non-inferential.

Full-set fingerprint: `sha256:fdf9bb90b9487b5931d48d5d4641cd003520cd20c3eda1a7a122a84b9982c16c`  
Diagnostic 16-pair fingerprint: `sha256:a03c09b941fc389bba790feff3f496b2e5e27070a360728d4c239cdcd3d7d0f8`

## Independent-seed confirmation

M0′ used model seed `2026091302` and self-play seed `2026091303`. The canonical lineage contains 4 × 128 = **512 self-play games**, **15,455** replay positions, one sample per newly generated position, batch size 64, and checkpoints M0′…M4′. Replay audit found **512/512 valid games**, zero technical games, and a consistent z perspective.

| Comparison | W/L/D | Mean pair score | 95% Hoeffding interval |
|---|---:|---:|---:|
| M1′ vs M0′ | 91/37/0 | 0.7109 | [0.5412, 0.8807] |
| M4′ vs M0′ | 124/4/0 | 0.9688 | [0.7990, 1.0000] |
| M4′ vs M1′ | 113/15/0 | 0.8828 | [0.7130, 1.0000] |

The M4′ vs M1′ result is the primary incremental-strength test. Adjacent progression diagnostics on the 16-pair subset were M2′ vs M1′ = 25/7/0 (0.7813), M3′ vs M2′ = 19/13/0 (0.5938), and M4′ vs M3′ = 18/14/0 (0.5625); all had zero technical games.

## Stage 3 retrospective

The six requested comparisons were rerun on the 16-pair diagnostic subset, with empty-board controls:

| Comparison | W/L/D | Mean pair score | 95% Hoeffding interval |
|---|---:|---:|---:|
| M1 vs M0 | 32/0/0 | 1.0000 | [0.6605, 1.0000] |
| M2 vs M1 | 19/13/0 | 0.5938 | [0.2542, 0.9333] |
| M3 vs M2 | 16/16/0 | 0.5000 | [0.1605, 0.8395] |
| M4 vs M3 | 20/12/0 | 0.6250 | [0.2855, 0.9645] |
| M4 vs M1 | 17/15/0 | 0.5313 | [0.1917, 0.8708] |
| M4 vs M0 | 31/1/0 | 0.9688 | [0.6292, 1.0000] |

## A/B/C optimizer-reuse ablation

All arms started from the same M0′ and used the same 96-game training split; the 32-game holdout was never used by the optimizer. A and B consumed the same 25,600 samples, while C consumed each of the 3,151 unique training positions once.

| Arm | Data rule | Samples | Reuse ratio | Holdout value CE |
|---|---|---:|---:|---:|
| A | first 16 games, high reuse | 25,600 | 58.05× | 0.8434 |
| B | all 96 games, high reuse | 25,600 | 8.12× | 0.5556 |
| C | all 96 games, one epoch without replacement | 3,151 | 1.00× | 0.6541 |

On the 16-pair ablation Arena, A vs M0′ was 29/3/0 (0.9063), B vs M0′ was 31/1/0 (0.9688), C vs M0′ was 27/5/0 (0.8438), and B vs C was 30/2/0 (0.9375). This supports unique-data volume as the stronger causal factor at the tested budget; the low-reuse-only claim remains inconclusive.

## Reproducibility and artifacts

Source commit: `706c1bb9a7bd063fa6e9265ab64f667df75f9c66`  
Source tree: `1ca84e1fdfed940fdcb71530f0b40580f52aca2e`  
Selected device: CPU (fixed batch-1 benchmark; CUDA was slower in this environment)  
Artifact root: `runs/torus-golden-stage4/torus-golden-stage4-seed2-v4/`

The run contains the frozen Evaluation V2 manifest, starts, all replay/self-play chunks, checkpoints and metadata, all Arena records and controls, ablation outputs, telemetry, and machine-readable `final-report.json`.

Limitation: the replay audit validates consistency against the same Golden rules implementation; it is not an independent rules oracle.
