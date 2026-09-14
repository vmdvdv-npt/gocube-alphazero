# Telemetry Schema: Historical Torus9/GoCube vs Post-PR93

Date: 2026-09-14

Purpose: compare **where a metric is computed and what it counts**, not merely field names. Old means the proven multiprocessing/central-inference architecture and its historical telemetry family; new means the post-PR93 current Torus9 continuation path.

## Required comparison

| OLD FIELD | NEW FIELD | OLD FORMULA | NEW FORMULA | SOURCE | SAME SEMANTICS? | ACTION |
|---|---|---|---|---|---|---|
| inference calls / batch count | `inference_calls` (`forward_calls`) | count of parent/broker NN forward invocations after cross-worker aggregation | number of `Torus9InferenceCoordinator` batches/forwards | old `training_common.py`/inference broker; new `torus9.py`, continuation summary | **PARTIAL** | retain canonical `inference_forward_calls`; document broker population and process scope |
| inference rows | `total_inference_rows` (`total_rows`) | sum of all worker rows included in central forwards | sum of requests handled by coordinator; normally one state row per pending thread lane | same | **PARTIAL** | canonicalize to number of network input rows, plus request count separately |
| rows/sec | `inference_rows_per_sec` | inference rows / self-play wall | `total_rows / selfplay_wall` | continuation summary | **YES formula; NO execution population** | keep formula; always pair with architecture ID/PID count |
| moves/sec | `moves_per_sec` | completed game moves/decisions / self-play wall | `sum(final_action_trace lengths) / selfplay_wall` | historical run summaries; continuation summary | **YES intent** | freeze exact definition as committed plies, excluding non-move NN calls |
| games/sec | `games_per_sec` | completed games / phase wall | `64 / selfplay_wall` | summaries | **YES** | use actual completed canonical game count, not hard-coded 64 in generic telemetry |
| mean batch rows | `mean_inference_batch_rows` | mean rows per parent central GPU forward across OS-worker requests | mean `batch_rows` per thread-coordinator GPU forward | old central broker; new coordinator | **NO architecture semantics** | retain value but add `batch_aggregation_scope=cross_process|cross_thread|in_game` |
| median/p50 batch rows | `median_inference_batch_rows` | p50 over central forward row counts where raw distribution existed | p50 over coordinator `batch_rows` | continuation helper | **PARTIAL** | canonical name `inference_batch_rows_p50` |
| p95 batch rows | `p95_inference_batch_rows` | p95 over central forward row counts where available | p95 over coordinator `batch_rows` | continuation helper | **PARTIAL** | canonical name `inference_batch_rows_p95` |
| max batch rows | `max_inference_batch_rows` | max rows in one broker forward | max coordinator batch rows | broker/coordinator | **PARTIAL** | keep; also persist configured cap and effective possible cap |
| configured coalescing wait | coordinator `wait_ms` / execution `wait_ms` | configured parent broker coalescing window | configured deadline after first request; Arena additionally sleeps before model batch | batching implementations | **NO** | split `batch_wait_config_ms` from observed queue/broker delay |
| observed queue wait | **missing** | queue-ready-to-broker-service latency was conceptually present but not consistently persisted | not measured | queue/broker | **NO / MISSING** | add mean/p50/p95/max observed queue wait, measured monotonic timestamps |
| queue depth | **missing** | ready-worker queue occupancy / pending requests; not consistently persisted | `queue.Queue` exists but depth is not telemetry | queue implementations | **NO / MISSING** | add sampled/peak pending request count and ready-worker count |
| batch cap hit ratio | **missing** | historical broker had physical/worker limits but no canonical ratio | explicit cap exists, no hit counter | batching implementations | **NO / MISSING** | add `cap_hit_batches / inference_forward_calls` and rows blocked by cap |
| active MCTS workers | `max_active_mcts_lanes` | active **OS worker processes** doing search | active Python thread lanes around `Torus9SelfPlayRunner` | process tree vs `Torus9ExecutionActivity` | **NO** | rename current to `active_mcts_thread_lanes`; add OS process telemetry |
| active inference requests | `max_active_inference_requests` | outstanding broker requests across process workers | outstanding coordinator requests across thread lanes | execution tracker | **NO architecture semantics** | add aggregation scope/process IDs |
| worker PID count | **missing** | number/set of live OS self-play/Arena worker PIDs | not reported | process tree | **NO / MISSING** | mandatory `worker_pid_count`, stable PID set sample, worker lifecycle events |
| effective CPU cores | **missing** | aggregate process CPU seconds / wall seconds | no direct metric | process/hardware telemetry | **NO / MISSING** | mandatory `effective_cpu_cores`; include parent+all children |
| CPU average | `cpu_utilization_pct` | historical hardware/process-tree sampling or aggregate CPU accounting | `100 * RUSAGE_SELF CPU seconds / (wall * os.cpu_count())` | historical hardware telemetry; continuation | **NO** | replace with process-tree aggregate; keep normalized percent and effective cores |
| CPU peak | **missing** | peak sampled CPU/process-tree utilization where hardware sampler used | absent | hardware telemetry | **NO / MISSING** | add avg, p95, peak for process tree and host |
| GPU average | `gpu_utilization_avg_pct` | nvidia-smi/device sampler average over phase | `GpuSampler` average over phase | old/new samplers | **MOSTLY** | freeze sample cadence/device identity and phase boundaries |
| GPU peak | `gpu_utilization_peak_pct` | peak sampled GPU util | sampler peak | same | **MOSTLY** | keep with cadence/source metadata |
| GPU idle duty | **missing** | not always persisted | not persisted | sampler | **MISSING** | add fraction of samples below idle threshold; needed for M18 gate |
| VRAM avg/peak | current peak field(s) | device memory samples/peak in hardware runs | PR93 records VRAM peak; average is not a guaranteed canonical field | sampler | **PARTIAL** | canonicalize avg/p95/peak allocated/used MiB |
| self-play wall | `self_play_wall_time_sec` | end-start monotonic self-play phase | `perf_counter` elapsed around self-play call | runner | **YES** | keep |
| training wall | `training_wall_time_sec` | end-start training phase | `perf_counter` elapsed around fixed-budget trainer | runner | **YES** | keep |
| total iteration wall | **missing canonical field** | iteration end-start where orchestration telemetry existed | not explicitly persisted in PR93 summary | orchestration | **NO / MISSING** | add `iteration_wall_time_sec`, with phase breakdown reconciliation |
| generated positions | `fresh_positions` | scientifically valid saved training positions from new games | length of fresh replay rows after technical-game exclusion | replay builder | **YES if named precisely** | use `fresh_valid_replay_positions`; do not call this samples consumed |
| replay positions | `replay.replay_positions` / checkpoint metadata | rows in replay window | rolling replay rows after last-3/cap policy | replay | **YES** | keep |
| training sample exposures | `training.samples_consumed` | optimizer updates x batch size, including reuse | 80 x 64 = 5120 | trainer | **YES** | keep distinct from unique rows; expose unique/reused counts |
| technical outcomes | `technical_games`, `technical_outcomes_excluded` | meaning changed across historical fixes; some old lines counted/handled technical events differently | record has technical termination; excluded from replay/scientific W/L | PR87 lineage/current runner | **NO historically; current contract is correct** | version termination schema; separate invalid, watchdog, exception, protocol/routing failures |
| lock scope | text telemetry only | broker/model/slot locks depended on old implementation | coordinator reports model-forward lock scope; activity tracker has small counter lock | batching code | **NO** | add architecture doc; optionally lock-wait telemetry if contention material |

## Current post-PR93 formulas that are safe to keep

- `inference_rows_per_sec = total_inference_rows / self_play_wall_time_sec`
- `games_per_sec = completed_games / self_play_wall_time_sec`
- `moves_per_sec = committed_plies / self_play_wall_time_sec`
- `mean/p50/p95/max_inference_batch_rows` over the raw list of GPU-forward row counts
- `training_wall_time_sec` as monotonic wall around fixed-budget optimization
- `fresh_valid_replay_positions` after exclusion of technical games
- `training_sample_exposures = optimizer_steps * batch_size`

The formulas above do not make current performance equivalent to the old architecture; architecture scope must be part of the schema.

## Fields required before M18

The replacement telemetry schema must carry a schema version and at minimum persist:

- architecture ID and batching scope (`cross_process`, not just `cross_thread`)
- configured worker count
- live/unique worker PID count and PID set/hash
- aggregate worker CPU seconds
- `effective_cpu_cores = aggregate_worker_cpu_seconds / self_play_wall_time_sec`
- process-tree CPU avg/p95/peak and host CPU avg/p95/peak
- inference forward calls and input rows
- rows/sec
- moves/sec and games/sec
- raw or histogrammed inference batch distribution; mean/p50/p95/max
- configured batch cap and wait
- actual queue-wait mean/p50/p95/max
- pending queue depth mean/p95/max
- cap-hit count and ratio
- active OS MCTS workers; thread lanes separately if used
- GPU avg/p95/peak, GPU idle-duty fraction and device identity
- VRAM avg/p95/peak
- self-play wall, training wall and total iteration wall
- fresh valid positions, replay positions, unique sampled rows, reused sample rows and total sample exposures
- explicit technical-outcome counters by category
- checkpoint/replay/iteration artifact identities sufficient to compare resume and non-resume runs

## Historical performance evidence to retain

Do not collapse unrelated runs into one synthetic baseline.

1. **Historical central-broker Legion run:** workers=16, many OS worker PIDs observed, mean inference batch approximately 60 rows, 256 games, self-play approximately 7m24s. Other exact fields are unknown from the currently located excerpt and must remain `unknown`, not inferred from another run.
2. **PR83 Golden Arena:** 16 OS workers, effective parallelism 13.492/16, 4.372x speedup, bit-exact 480/480 parity.
3. **PR93 M1 degraded current run:** 64 games, current 80x8 model, 64 sims, self-play 2260.36 s, 4993 moves, 2.2089 moves/s, 315477 inference rows/calls, 139.57 rows/s, mean/p50/p95 batch 1, GPU avg 34.87%, peak 79%, VRAM peak ~1098 MiB.

These baselines serve different purposes. #1 proves central cross-process batching existed. #2 proves Golden process parallelism existed without semantic drift. #3 is the regression floor that the restored path must beat materially.

## Acceptance semantics

For the M18 gate, a run is performance-degraded if `mean_inference_batch_rows < 16`. Do not repeat a long Arena with the same settings in that state.

`workers=16` passes only when OS-process telemetry proves actual process-level workers. Sixteen logical lanes or 16 Python threads is not equivalent.
