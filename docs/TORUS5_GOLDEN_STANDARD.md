# Torus 5x5 Golden Standard

## Canonical identity

The only current entry in the Golden Standard catalog is:

```text
gocube-torus5-golden-current -> gocube-torus5-golden-v2
```

The concrete v2 profile is stored in
[`configs/gocube/torus5_golden_v2.json`](../configs/gocube/torus5_golden_v2.json).
The alias and catalog are stored in
[`configs/gocube/torus5_golden_current.json`](../configs/gocube/torus5_golden_current.json)
and [`configs/gocube/golden_standards.json`](../configs/gocube/golden_standards.json).

Resolve and smoke-test the current standard with:

```bash
python -m tools.torus5_golden --run-name torus5-v2-smoke --smoke
```

This writes an immutable `run-manifest.json` containing the requested alias,
concrete preset version, resolved configuration and fingerprint, network
identity, komi, and git commit/tree identity. It performs one model forward
pass and one tiny optimizer step only; it does not start a long training or
Arena run.

## Universal versus board-specific settings

[`configs/gocube/universal_training_v1.json`](../configs/gocube/universal_training_v1.json)
contains the settings extracted from the current Torus 9x9 stable-learning
profile. Resolution verifies that the source Torus9 fingerprint and every
projected universal value still match, so a Torus9 change cannot silently
change the Torus5 standard.

Universal settings include:

- sequential 64-simulation PUCT, `cpuct=1.25`, `fpu=0.0`;
- self-play root noise (`epsilon=0.25`, `alpha=0.30`), temperature 1.0 on
  plies 1--8 and 0 afterwards, with fast search and resign disabled;
- noise-free deterministic Arena at temperature 0;
- batch-one, non-coalesced inference on the stable-learning path;
- Adam (`lr=0.001`, `weight_decay=0`), batch 64, 80 optimizer steps and
  5,120 samples per iteration;
- three-generation rolling replay capped at 20,000 positions;
- 64 games per iteration and 16 workers.

Board-specific settings remain in v2: standalone Torus 5x5 topology, graph
area rules, a 500-action watchdog, 5x5 evaluation starts, action space 25
points plus PASS, and the network `48 channels x 6 blocks`. The network has
WDL and policy heads only (`[26]` and `[3]`), and the standard komi is exactly
`0.5`.

## Legacy and historical profiles

The following profiles remain available for reproducibility but are deprecated:

- `gocube-torus5-golden-v1-legacy` -> `torus_golden_training_v1.json`;
- `gocube-torus5-golden-stage2-v2-legacy` -> `torus_golden_v2.json`;
- `gocube-torus5-golden-stage4-v2-legacy` ->
  `torus_golden_training_v2_data_rich.json`.

They cannot be selected through the default/current resolver. Reproduction
requires an explicit override:

```bash
python -m tools.torus5_golden \
  --preset gocube-torus5-golden-v1-legacy \
  --allow-legacy-config \
  --run-name torus5-v1-reproduction \
  --smoke
```

The historical Stage-3 and Stage-4 runners are also guarded. They require
`--allow-legacy-config` and are not current launch paths.
