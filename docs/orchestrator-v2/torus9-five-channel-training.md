# Ordinary Torus9 training after komi adaptation

The existing `continuous` V2 coordinator now selects the ordinary 5-channel
production driver. Its default Torus9 generation bridge rejects legacy 6-channel
configs before self-play or training. Historical six-channel files and conversion
code remain read-only provenance: the 5CH bootstrap graph references M137.
Do not delete those ancestors or relabel the adaptation graph generations.

The first ordinary parent is the completed adaptation `update-2400.pt`, published
as internal generation 198. `M138-5CH` is its release alias, not a rewritten graph
identity. Five ordinary iterations therefore produce internal M199 through M203.
The final arena compares M203 with that exact parent.

Approved first block:

| Setting | Value |
| --- | --- |
| Network / referee komi | 5 channels / 1.5 |
| Generations / arena cadence / reference gap | 5 / 5 / 5 |
| Self-play | 384 games per iteration, 200 simulations |
| Training | Adam, LR 0.00005, 160 updates, batch 64 |
| Regularization | No L2-SP, no weight decay, gradient norm clip 1 |
| Replay | Last six generation buckets, no position cap |
| Arena | 192 games, 128 simulations, paired color-swapped starts |
| Arena search | cpuct 1.25, FPU 0, no noise, no resign, temperature 0 |

The effective config selects `extensions.training_driver =
"torus9-five-channel-ordinary-training-v1"`. It binds the complete
`adaptation_parent` CheckpointRef, `initial_replay_buckets` (all shards referenced
by the parent checkpoint), and `validation_buckets`. Shards use absolute storage
paths with SHA-256 identities; every payload is verified before use. The first
new generation replaces the oldest inherited bucket, preserving a six-bucket
window. Validation games remain excluded. Ordinary sampling is uniform over
positions; the validation sampler is kept consistent with adaptation.

Each named Adam state is loaded unchanged. Its inherited clock is retained
independently; every parameter advances by one per ordinary update. No bias
moment reset, freeze schedule, or anchor penalty occurs. Self-play uses the
immediate parent, so each generation receives fresh games from the latest model.

The ordinary driver runs only inside the existing V2 generation child permit.
It uses the cooperative self-play engine, standard graph/catalog publication,
completion fence, immutable Git runtime, supervisor and ArenaRunnerV2. Collected
128-game shards are resumable and bound to parent/config identities. Training
restarts deterministically from the generation parent if interrupted before
commit. There is no additional execution authority issuer or standalone launcher.

On completion, `reports/block-report.{json,md}` records the five iterations,
final checkpoint and arena result. The finite coordinator budget stops the run;
no second block or LR increase is automatic. The 128-simulation arena is a new
evaluation setting and should not be directly conflated with the adaptation's
256-simulation results.
