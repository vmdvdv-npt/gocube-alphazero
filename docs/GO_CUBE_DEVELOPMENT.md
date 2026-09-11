# GoCube development launcher

For normal local GoCube ↔ AlphaZero replay work, start the integration service from the `gocube-alphazero` checkout with one command:

```bash
./dev-gocube
```

The launcher is intentionally for generated AI self-play games and replay/diagnostics. It does not add a human-vs-AI mode and it does not control training.

## What the launcher does

`./dev-gocube`:

1. resolves the repository root, so it works regardless of the shell's current directory;
2. uses the repository's existing `.venv/bin/python` directly, so manual virtualenv activation is not required;
3. verifies that the local `checkpoint/` directory exists;
4. idempotently registers the two known legacy runs when they are present:
   - `gocube-cube4-stage4-v1` — Cube 4×4, Chinese rules, komi 7.5;
   - `torus-9x9-30iter` — Torus 9×9, Chinese rules, komi 7.5;
5. leaves compatible existing manifests unchanged and refuses to overwrite an incompatible manifest;
6. fails clearly if `127.0.0.1:8765` is already occupied instead of silently starting another service;
7. runs the existing Protocol V1 service on `http://127.0.0.1:8765` with `device=auto`.

New training runs already write their own GoCube manifest, so they do not need to be added to the legacy bootstrap list.

Once the launcher is running, use GoCube's Development Workspace to generate and replay games. Repeated test games should be generated from the UI; `curl`, manual checkpoint IDs, JSON payloads, and repeated manifest registration are not part of the normal workflow.

Stop the service with `Ctrl+C`. Because the shell launcher uses `exec`, the service receives terminal signals directly.

## Optional overrides

The defaults match GoCube Development Workspace. Advanced local use can override the service parameters, for example:

```bash
./dev-gocube --device cpu
./dev-gocube --port 8877
```

If the port is changed, GoCube's configured AlphaZero base URL must be changed to match.
