# Training reports

This branch is a machine-written transport for compact AlphaZero training diagnostics.

For each run, the reporter stores only:
- `run.md` — run status and launch command;
- `console-tail.log` — a bounded tail of console output;
- `manifests/*.json` — completed iteration manifests.

Full console logs, checkpoints, training data, and per-game JSON records remain local on the training machine.
