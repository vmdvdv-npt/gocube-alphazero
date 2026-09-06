# Training run

- Run: c4-hparam-night-20260906-005747
- Status: INTERRUPTED
- Started: 2026-09-06T09:20:48+04:00
- Ended: 2026-09-06T10:51:43+04:00
- Exit code: 130
- Source commit: 85c87a7cfd467a4d3f4b2844253fb63d746d672a
- Completed iteration manifests: 0
- Structured experiment artifacts: yes
- Reports branch: training-reports

## Command

```bash
/home/codex/projects/gocube-alphazero/.venv/bin/python /home/codex/.cache/gocube-night-tools-028342cee09a757a8c9c84b225c550c36bbf601a/c4_adaptive_finish.py --experiment-id c4-hparam-night-20260906-005747 --source-run c4-t001-c4-c001 --frozen-commit 85c87a7cfd467a4d3f4b2844253fb63d746d672a --max-hours 999 --adaptive-block-games 24 --adaptive-max-games 72 --adaptive-final-block-games 32 --adaptive-final-max-games 128
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
