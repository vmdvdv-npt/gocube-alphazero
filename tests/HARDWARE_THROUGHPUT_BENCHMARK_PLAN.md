# Legion hardware throughput benchmark plan

Purpose: optimize the **execution configuration of the training machine**, separately from AlphaZero hyperparameter tuning.

This benchmark is intended for future Cube/Torus training runs. It must answer a different question from `sims / pFast / games-per-iteration` experiments:

> Which workers/batching configuration produces the highest **sustained useful training throughput** on this PC without losing performance to the shared CPU/GPU power and cooling budget?

## Why total power matters

The machine may not be able to sustain simultaneous maximum CPU and GPU package power because CPU and GPU share chassis airflow, cooling capacity, power delivery, and ambient thermal headroom.

Therefore:

- do **not** optimize for maximum CPU utilization alone;
- do **not** optimize for maximum GPU utilization alone;
- do **not** assume `CPU 90% + GPU 30%` is inefficient;
- compare configurations only by **sustained throughput after thermal equilibrium**;
- record total wall power whenever practical, because CPU package power + GPU board power does not include RAM, VRM, motherboard, storage, fans, PSU losses, etc.

A configuration with lower instantaneous utilization may be faster if it avoids thermal/power throttling and sustains higher clocks.

## Keep algorithmic variables fixed

Hardware benchmark runs must use the same:

- topology and board size;
- checkpoint/model;
- ruleset and komi;
- regular MCTS sims;
- fast MCTS sims;
- `pFast`;
- games per measured workload unless that exact variable is the infrastructure variable under test;
- train batch size;
- endgame weighting;
- seeds where deterministic/reasonable;
- network architecture and checkpoint;
- Arena disabled unless the benchmark is specifically measuring Arena throughput.

The benchmark is invalid if algorithmic strength parameters change together with execution parameters.

## Infrastructure variables to test

Start with a small matrix rather than a huge grid.

Suggested first pass:

| Config | Workers | Process batch / games geometry | Inference batch wait |
|---|---:|---|---:|
| H1 | 12 | baseline-compatible | 1 ms |
| H2 | 14 | baseline-compatible | 1 ms |
| H3 | 16 | baseline-compatible | 1 ms |
| H4 | best workers from H1-H3 | larger/coalesced batch candidate | 1 ms |
| H5 | best workers/batch | same batch | 0 ms |
| H6 | best workers/batch | same batch | 2-3 ms |

Do not change more than one infrastructure axis at a time unless explicitly evaluating a practical package.

## Sampling interval

Record resource telemetry every **5 seconds** by default. Ten seconds is acceptable for very long runs, but five seconds is preferred for detecting clocks/throttling transitions.

Every sample must contain an absolute timestamp so it can be joined with training logs.

## CPU telemetry

Record when available:

- total CPU utilization;
- per-logical-core utilization;
- `train.py` CPU utilization;
- CPU package power;
- CPU package temperature;
- effective CPU clock / average effective clock;
- per-core or package effective clocks if available;
- CPU thermal throttling flag/reason;
- CPU power/current-limit throttling flag/reason;
- load average;
- process/thread count if useful for diagnosing contention.

Under WSL, package power and throttling data may be unavailable or unreliable. In that case collect it from the Windows side (for example HWiNFO logging) and join by timestamp.

## GPU telemetry

For NVIDIA hardware record when available:

- GPU utilization;
- memory-controller utilization if available;
- VRAM used/total;
- GPU board power draw;
- GPU power limit;
- GPU temperature;
- GPU hotspot temperature if available;
- graphics/SM clock;
- memory clock;
- fan speed/RPM if available;
- performance state;
- thermal/power/voltage utilization-limit reasons if exposed.

`nvidia-smi` telemetry is sufficient for the Linux/WSL side when the relevant fields are exposed.

## Memory/system telemetry

Record:

- RAM used/available;
- swap used;
- VRAM used;
- disk read/write rate if sample/checkpoint writes appear to stall;
- optional WSL memory pressure indicators.

## Total system power

Preferred measurement: **wall power** from a logging wattmeter/smart plug, or another source that can export timestamped power samples.

If available, record:

- instantaneous wall watts;
- average wall watts during the steady-state measurement window;
- peak wall watts;
- watt-hours consumed during the measured window.

If no external wall meter is available, explicitly mark wall power as unavailable rather than estimating it from CPU+GPU package power.

## Cooling telemetry

Record when practical:

- CPU fan RPM;
- GPU fan RPM;
- case fan RPM;
- ambient/room temperature at benchmark start;
- chassis thermal mode / fan profile;
- whether the case, vents, filters, or laptop/desktop placement changed.

Cooling configuration must remain constant across candidates.

## Training phase markers

Resource utilization must be interpreted by **phase**, not only as one run-wide average.

The telemetry/log join should identify at least:

- self-play;
- inference during self-play;
- sample serialization/saving;
- optimizer training;
- checkpoint saving;
- Arena evaluation, when benchmarked separately.

Report CPU/GPU/power statistics separately for each phase.

A run-wide `GPU utilization = 30%` is not enough to diagnose a bottleneck.

## Useful-work metrics

Record the useful output of the machine, including as many of these as the current instrumentation exposes:

### Self-play

- games completed;
- games/hour;
- positions generated;
- positions/second;
- MCTS decisions/second;
- regular/fast decision counts;
- average game length;
- inference batches/second;
- average inference batch size;
- inference rows/second;
- wall seconds for self-play.

### Optimizer

- samples/examples processed;
- optimizer steps;
- examples/second;
- optimizer steps/second;
- wall seconds for training.

### End-to-end

- complete iteration wall time;
- samples generated/hour;
- useful training examples/hour;
- checkpoint-to-checkpoint elapsed time.

## Thermal steady-state protocol

Short cold-start measurements are not sufficient.

For every candidate configuration:

1. start from the same software/checkpoint conditions;
2. allow the machine to warm up;
3. run long enough for CPU/GPU temperature and effective clocks to stabilize;
4. use at least **15-30 minutes of sustained load** for judgment;
5. preferably exclude the first ~10 minutes from steady-state averages when the run is long enough;
6. note any clock decline after warm-up;
7. reject conclusions based only on the first few minutes.

If practical, repeat the leading configurations twice to make sure the result is not run-order or ambient-temperature noise.

## Primary selection criterion

The objective is **maximum sustained useful throughput**, not maximum utilization.

Primary metric for self-play benchmarking:

> sustained positions/second (or games/hour when positions/second is unavailable)

Secondary metrics:

- full iteration wall time;
- inference rows/second;
- examples/second during optimizer training;
- thermal stability;
- wall watts;
- watt-hours per useful output.

When wall-energy data exists, also compute:

- games/kWh;
- positions/kWh;
- training examples/kWh;
- complete iterations/kWh if meaningful.

## Throttling rules

Flag a configuration as thermally/power constrained if any of these are observed after warm-up:

- effective CPU clocks decline materially while utilization remains high;
- GPU clocks decline materially under a stable workload;
- explicit thermal/power throttle flags appear;
- throughput declines during a long steady workload;
- increasing workers or batch pressure increases watts/temperature but not useful throughput;
- higher GPU utilization is accompanied by lower end-to-end throughput because CPU clocks collapse, or vice versa.

A throttling flag does not automatically disqualify a configuration if it still has the highest stable throughput, but the behavior must be reported explicitly.

## Result table

The final report should contain at least:

| Config | Workers | Batch/wait | CPU % | CPU W | CPU temp | CPU eff. clock | GPU % | GPU W | GPU temp | Wall W | Throttle | Games/h | Positions/s | Iter wall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|

Use steady-state averages plus relevant peaks/minima. Do not mix cold-start and steady-state values silently.

## Interpretation examples

Do not prefer a configuration just because it has a visually higher utilization number.

Example:

- 16 workers: CPU 94%, GPU 30%, 330 W wall, 450 games/hour;
- 14 workers + better batching: CPU 82%, GPU 55%, 345 W wall, 520 games/hour.

The second configuration is better despite lower CPU utilization because it produces more useful work per hour.

Likewise, if a configuration raises both CPU and GPU utilization but causes sustained clocks to fall and throughput to decline, it is worse.

## Relationship to AlphaZero hyperparameter tuning

Keep two experiments conceptually separate:

1. **Algorithm tuning**: `sims`, `pFast`, games/iteration, learning/training choices, playing strength per compute.
2. **Hardware/execution tuning**: workers, inference batching, waits, CPU/GPU balance, thermal/power behavior, throughput per wall-clock/energy.

Run hardware tuning once per materially different machine/cooling/software environment and reuse the result across Cube 2x2, 3x3, 4x4, larger Cube sizes, and Torus where the execution path is comparable.

Re-run the benchmark when hardware, cooling, drivers, PyTorch/CUDA, WSL behavior, inference implementation, network architecture, or batching implementation changes materially.

## Future implementation target

When automated tooling is added, it should produce machine-readable artifacts such as:

- `hardware-samples.csv` — timestamped 5-second telemetry;
- `phase-events.jsonl` — self-play/train/save/Arena transitions;
- `throughput.csv` — useful-work counters over time;
- `hardware-summary.json` — steady-state aggregates and throttling flags;
- `hardware-summary.md` — human-readable comparison/recommendation.

The benchmark runner should never silently claim unavailable sensors. Every missing field must be null/`unavailable` with its collection source recorded.
