# Cube 4 overnight hyperparameter experiment

- Experiment: `c4-hparam-night-20260906-005023`
- Status: `RUNNING`
- Started: 2026-09-06T00:51:39+04:00
- Deadline: 2026-09-06T08:51:39+04:00
- Frozen training commit: `85c87a7cfd467a4d3f4b2844253fb63d746d672a`
- Parent: `c4-t001-c4-c001@5`
- Parent checkpoint SHA256: `pending`
- Workers: **16** everywhere

## Decision status

No strength evaluation has completed yet; throughput/training diagnostics alone are insufficient to choose a winner.

Primary ranking is based on fixed-Arena playing strength, with wall-clock cost shown alongside it. Training losses are diagnostics only.

## Branch design

| Branch | Regular sims | pFast | Games/iteration | Process batch | Purpose |
|---|---:|---:|---:|---:|---|
| A | 100 | 0.25 | 256 | 16 | baseline |
| B | 100 | 0.00 | 256 | 16 | no fast search |
| C | 100 | 0.50 | 256 | 16 | more fast search |
| D | 50 | 0.25 | 256 | 16 | shallower regular search |
| E | 200 | 0.25 | 256 | 16 | deeper regular search |
| F | 100 | 0.25 | 128 | 8 | faster feedback loop |
| G | 100 | 0.25 | 512 | 32 | more data per update |

> **Causal caveat for F/G:** in the current implementation `process_batch_size = games_per_iteration / workers`. Therefore 128/256/512 games also means process batches 8/16/32. F/G measure the practical outer-loop package, not a perfectly isolated game-count variable.

## Latest training state

| Branch | Latest iter | Train wall | Samples (latest) | Opt steps | Realized fast | No-result | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 5 | 0.00 h | — | — | — | — | PENDING |
| B | 5 | 0.00 h | — | — | — | — | PENDING |
| C | 5 | 0.00 h | — | — | — | — | PENDING |
| D | 5 | 0.00 h | — | — | — | — | PENDING |
| E | 5 | 0.00 h | — | — | — | — | PENDING |
| F | 5 | 0.00 h | — | — | — | — | PENDING |
| G | 5 | 0.00 h | — | — | — | — | PENDING |

## Strength evaluations

| Candidate | Reference | Games | W-L-D-NR | Score | Approx. 95% CI | Eval wall |
|---|---|---:|---|---:|---|---:|
| — | — | — | — | — | — | — |

## Axis readout after the first descendant iteration

- **pfast:** B=pending, C=pending.
- **sims:** D=pending, E=pending.
- **games:** F=pending, G=pending.

## Report completeness

The GitHub report intentionally includes `experiment.json`, `state.json`, `metrics.csv`, `evaluations.csv`, `artifacts.json`, `events.jsonl`, compact iteration manifests, evaluation JSON files, and per-stage log tails. Together these preserve exact configuration/provenance, compute cost, self-play mix, data volume, optimizer work, terminal quality, checkpoint hashes, and fixed-Arena strength.

A final choice should not be made from loss curves alone. Prefer direct same-depth comparison against A; if its interval overlaps 0.5, repeat the leading candidate with another self-play/evaluation seed before committing to long training.
