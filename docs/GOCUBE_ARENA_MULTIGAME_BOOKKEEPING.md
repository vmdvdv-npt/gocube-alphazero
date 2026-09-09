# GoCube Arena multi-game bookkeeping

## Root cause

The old checkpoint Arena used one mutable `player_to_index` mapping per
worker.  `SelfPlayAgent` could hold several game slots, but a result only
carried `(final_state, winstate, worker_id)`.  If slot 0 finished, was
recycled, and changed color before the parent drained the queue, the parent
interpreted the old result with the recycled worker mapping.  The same
ambiguity applied to coalesced search rows: the parent could split network
outputs by worker-sized chunks, but had no token proving that a returned row
belonged to the slot generation that requested it.  The defensive
`arena_batch_size = 1` setting avoided this historical failure by ensuring a
worker had only one active game.

## Identity and routing model

Every requested game receives a global integer `game_id`.  IDs are partitioned
into fixed per-worker quotas before workers start.  The color schedule is
global and deterministic: even IDs are `B0 black / B1 white`; odd IDs are
`B0 white / B1 black`.  Each active slot has an immutable
`ArenaGameIdentity(game_id, worker_id, slot_id, generation, model_a_color,
player_to_index)`, plus its game state, the tuple of per-model MCTS trees, and
completion state.

Each inference payload is grouped by model/network and carries one
`ArenaRoutingKey(worker_id, slot_id, generation, game_id)` for every tensor
row.  The parent coalescer forwards B0 and B1 rows separately, preserving the
18-channel and 20-channel observation adapters.  The worker receives the
ordered routing-key list with the response and rejects a response whose list
does not exactly match the current slot-generation request.

Completed results are `ArenaResult` snapshots.  Parent attribution uses the
snapshot's `player_to_index` and never reads a worker's current mutable color.
The parent also rejects duplicate IDs and verifies the complete ID set is
`0..games-1`.

## Fixed search contract

The multi-game path keeps Cube 4x4, Japanese rules, `komi=0.5`, 50 Arena
simulations, fast simulations off, root noise off, root temperature off, and
move temperature 0.  B0/B1 model forwards remain separate four-head forwards
for policy, value, score, and ownership.

## Benchmark

Benchmark command contract: the real iteration-0016 B0/B1 checkpoints, CUDA,
16 workers, 50 sims, fast/noise/temperature disabled, and the same game count
for every batch size.  GPU/CPU telemetry is reported as unavailable when the
host sensor is not exposed.

| arena_batch_size | games | workers | wall time (s) | mean inference rows | games/sec | peak VRAM (MiB) | correctness |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 (old baseline) | 256 | 16 | 3805.63 | 4.303 | 0.06727 | 12.86 | 256 results; pre-fix worker bookkeeping |
| 1 | 32 | 16 | 474.23 | 4.753 | 0.06748 | 13.42 | 32 unique IDs |
| 2 | 8 | 2 | 405.49 | 1.112 | 0.01973 | 10.26 | 8 unique IDs |
| 4 | 32 | 16 | 541.38 | 6.091 | 0.05911 | 13.42 | 32 unique IDs |
| 4 | 8 | 2 | 256.21 | 1.807 | 0.03122 | 10.30 | 8 unique IDs |
| 8 | 8 | 2 | 253.81 | 1.797 | 0.03152 | 10.30 | 8 unique IDs |

The production default is `--arena-batch-size 4`; it is configurable and is
selected as the smallest setting that showed the best throughput in the
multi-game controlled sample.  Batch 8 did not improve on batch 4, while the
16-worker/32-game sample is CPU/control-path bound and underfills a batch-4
worker (two games per worker).  The host did not expose `nvidia-smi`, so GPU
utilization, CPU utilization, and RAM percentiles are recorded as unavailable;
peak CUDA allocation and correctness were still collected.
