# External training efficiency monitor

Use `tools/training_efficiency_monitor.py` for a passive measurement of **one
upcoming generation** of Torus9 ordinary 5CH training. It reads existing V2
heartbeat/artifacts and NVIDIA telemetry. It does not import training code,
launch/stop a job, change configuration, send notifications, or write into the
run. The module uses Python's standard library; no installation is required.

```sh
python3 tools/training_efficiency_monitor.py \
  --run-root /absolute/path/to/runs/torus9/active/LINEAGE \
  --parameters /absolute/path/to/runs/torus9/orchestration/jobs/LINEAGE/parameters.json \
  --generation 254 \
  --output /absolute/path/outside-the-run/measurement-254
```

Start before self-play finishes to capture the entire training phase. Default
sampling is every second; `--interval` changes it. It stops after the generation
commit, a dead generation process, or `--max-hours` (default 24). For detached
collection use `nohup python3 ... > monitor.log 2>&1 < /dev/null &`; verify the
PID in `monitor.json` and growing `samples.jsonl`. Interrupting **only this
monitor's PID** does not signal the trainer. Each output directory is exclusive.
The monitor does not notify the user automatically; saved files remain usable
when the chat is closed.

Outputs:

- `samples.jsonl`: timestamped raw GPU readings, heartbeat progress and trainer
  PID CPU time/RSS/thread count. GPU samples include load, used/total VRAM,
  memory-controller activity, temperature, power, clocks and P-state. Errors and
  unavailable readings are recorded explicitly. All visible GPUs are sampled.
- `report.json`: refreshed every 15 seconds and finalized at exit. Records
  optimizer updates, batch draws, updates/minute, examples/minute, sampled GPU
  statistics, generation duration and training-through-commit duration.
- `training-metrics.json`: a separate copy of the completed driver's loss,
  gradient and validation metrics, if published. No checkpoint is loaded.
- `monitor.json`: monitor PID, source, sampling settings and operator parameters.

The current driver has **no heartbeat at optimizer-loop entry**. Start time is
therefore the rolling replay manifest's filesystem modification time, published
immediately before the update loop. This includes a small setup gap, and can be
affected by clock adjustments or later copying of files. End time comes from
`progress_at` of the final training update. If that update is missed between
polls, exact loop duration/rates remain null; raw samples preserve the observed
progress. Whole training-through-commit duration still measures replay manifest
publication through validation, checkpoint save/reload and generation commit.
Whole generation also includes preparation, self-play and replay loading;
subsequent Arena is excluded. Attach before loop entry for full GPU coverage.

GPU values are device-wide, so other workloads affect them. A sampled VRAM peak
is not a CUDA allocator peak. Means are sample averages. CPU time is for the
trainer PID only, not its self-play workers. Examples/minute counts repeated
batch draws, not unique examples or distinct parameters: each optimizer update
updates the model's trainable parameters. These observations can identify time
spent on training and GPU underuse; a causal batch 64/128 comparison requires
comparable workloads and equivalent sample/update budgets.
