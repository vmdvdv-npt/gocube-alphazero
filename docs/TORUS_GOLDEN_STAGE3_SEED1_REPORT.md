# STAGE 3 GOLDEN TORUS NEURAL PROOF

Base branch: `codex/torus-rebuild-v1`

Base SHA: `a3f151330aecec16e83764376496da68f96d92fe`

Feature branch: `codex/torus-rebuild-v1-stage3-neural-proof`

Training source SHA: `3c42aecda53cabfe4f4302027a9e6bc485ad9898`

Git tree: `5ce15c79f4f4eb0a5e6c9c4d4d8e5595f0b18d06`

PR: pending

Training profile: `gocube-torus-golden-training-v1`

Training profile fingerprint: `sha256:e7ba86167f549d07ff05cb010c0e52fcf898a66f3daea77e74c46eac872bf758`

Device: CPU (`torch 2.4.1+cu124`; CUDA was available but batch-1 preflight was materially slower for this tiny graph)

Model: `GoldenGraphNetV1`, 55,493 parameters, hidden 64, 4 graph residual blocks

Komi: `0.5`

Rules fingerprint: `sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842`

Topology fingerprint: `sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef`

Observation fingerprint: `sha256:d6e3aecc89f7df84f6e758423da4e3fe9269abeca644070db0be9261b30c6361`

Target fingerprint: `sha256:6dffad74c4f832741f9cd522aa407f3e88d6f6159ebe56c3dc226a52b4145e41`

Self-play contract: `golden-selfplay-search-v1`, 64 simulations, cpuct 1.25, FPU 0, root Dirichlet epsilon 0.25 / alpha 0.30, temperature 1.0 through ply 8 then 0.0, watchdog 500.

Arena contract: `golden-arena-search-v1`, 64 simulations, cpuct 1.25, FPU 0, noise OFF, fast search OFF, resign OFF, temperature 0.

Performance configuration: 16 independent game workers, one shared immutable model, inference batch size 1, inference coalescing disabled. The serial-vs-parallel equivalence gate passed for two fixed game IDs before chunk 1.

## Self-play and training

| Chunk | Source | Valid games | Technical | New positions | Cumulative positions | Optimizer updates | Policy loss | Value loss | Total loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | M0 | 16 | 0 | 497 | 497 | 200 | 2.232667 | 0.512722 | 2.745389 |
| 2 | M1 | 16 | 0 | 439 | 936 | 400 | 2.257115 | 0.439777 | 2.696892 |
| 3 | M2 | 16 | 0 | 557 | 1493 | 600 | 2.164865 | 0.416014 | 2.580879 |
| 4 | M3 | 16 | 0 | 562 | 2055 | 800 | 2.086270 | 0.277407 | 2.363677 |

Every chunk passed model-source hash validation. Every optimizer update passed finite loss, gradient, and parameter checks. All five model hashes differ.

## Checkpoint lineage

| Checkpoint | Model hash | Artifact SHA256 |
|---|---|---|
| M0 | `sha256:31ab64bc82f6de69c0b827fcf478d0a1adb8b54612987b8b60d9ac969ce81e00` | `sha256:577a6487658a2d5a0c1b105a79c5f132461b28b2f70d7e76cc9acbc4c853f793` |
| M1 | `sha256:c89394a6d61035fcdb58ea6bd4f0c61e2955cdce35ef85a50e16b2a4d53f7dbe` | `sha256:60a8e8895ee0c60758208bc2bf2488f8afab42393586cafd2c8a109ecac5d176` |
| M2 | `sha256:f3bfea86bc1c1acde4e3fcd2095b767e29adff8fcf2b34fe54886d7a9d4eef34` | `sha256:edc3457e61cc816c85f490aa8c1864054172ca6f22e6e264654b68a054dec205` |
| M3 | `sha256:afee33da702de87ae6bf64ace380271ed915ae8eae61f0d2862f25291d7ddec6` | `sha256:939520f709d8e55b7f8da74d8832a04a1c1dd3d5f0eb88e2acd3a69599c33b13` |
| M4 | `sha256:9518a60a9871e8593935ff0fb79ee57bf7494e76f03a084ffda288431fdd8b61` | `sha256:00dca066e6b8cea9e04ec41da6a3198b2f4efa17e8b1a30031e8f3370207a273` |

## Arena

The pre-generated start-set fingerprint was
`sha256:226e080c8308bb147946afe9de5db120572e6bf5a74d1cbb4ebfaa7e4092dcf5`.
Each comparison used 8 paired starts / 16 games, with the empty-board start
and legal prefix starts included. Technical Arena games: 0.

### M4 vs M1

- Pair score: `0.500`
- W/L/D: `8 / 8 / 0`
- M4 as Black: `8` wins; M4 as White: `0` wins
- 95% pair-score interval: `[0.500, 0.500]`
- Important diagnostic: M4's result was color-asymmetric in this small sample and did not exceed M1.

### M4 vs M0

- Pair score: `0.875`
- W/L/D: `14 / 2 / 0`
- M4 as Black: `8` wins; M4 as White: `6` wins
- 95% pair-score interval: `[0.715, 1.000]`

## Verdict

`LEARNING NOT DEMONSTRATED`

The pipeline, lineage, provenance, value perspective, replay construction, and
technical-failure gates are valid. M4 shows a strong diagnostic result against
M0, but the primary pre-declared comparison M4 vs M1 is exactly neutral, so
Stage 3 does not claim a positive learning proof.

Canonical artifact directory:
`runs/torus-golden-stage3/torus-golden-stage3-seed1-v5/`

