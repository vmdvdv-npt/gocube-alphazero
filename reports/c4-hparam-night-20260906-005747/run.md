# Training run

- Run: c4-hparam-night-20260906-005747
- Status: RUNNING
- Started: 2026-09-06T01:24:33+04:00
- Ended: —
- Exit code: —
- Source commit: 85c87a7cfd467a4d3f4b2844253fb63d746d672a
- Completed iteration manifests: 0
- Structured experiment artifacts: yes
- Reports branch: training-reports

## Command

```bash
/home/codex/projects/gocube-alphazero/.venv/bin/python /home/codex/.cache/gocube-night-tools-2a8e15c796a8390aae6ed68c13c3688fcd1383eb/c4_overnight_experiment.py --experiment-id c4-hparam-night-20260906-005747 --source-run c4-t001-c4-c001 --frozen-commit 85c87a7cfd467a4d3f4b2844253fb63d746d672a --max-hours 999
```

## Published to GitHub

- run.md — wrapper status summary
- console-tail.log — last 2000 console lines
- manifests/ — compact completed manifests for a normal training run
- structured files from `training_reports/c4-hparam-night-20260906-005747/publish/`, when present

For overnight experiments, start with `summary.md`, then inspect `metrics.csv`,
`evaluations.csv`, `experiment.json`, `state.json`, `artifacts.json`,
`events.jsonl`, and per-stage log tails.

Full wrapper console output remains local at `training_reports/c4-hparam-night-20260906-005747/console.log`.
