# GoCube development launcher

For normal local GoCube ↔ AlphaZero replay work, start the integration service from the `gocube-alphazero` checkout with one command:

```bash
./dev-gocube
```

The launcher is intentionally for generated AI self-play games and replay/diagnostics. It does not add a human-vs-AI mode and it does not control training.

## Komi and legacy checkpoints

The current project-wide policy is `docs/KOMI_POLICY.md`: `0.5` is the default
baseline, but it is not globally hard-coded as the only future-valid Torus
komi. Legacy `7.5` is forbidden in current runtime paths and is treated as
likely stale-artifact contamination requiring owner review.

Older revisions of this document associated the names
`gocube-cube4-stage4-v1` and `torus-9x9-30iter` with legacy `7.5` semantics,
while later launcher code attempted to register those names as `0.5`. Because
the pre-manifest artifacts do not provide a trustworthy basis for silently
choosing between those meanings, automatic registration of those historical
run names is now disabled.

If one of those checkpoints is needed for historical analysis, audit its
actual provenance first and register it explicitly. If the evidence indicates
`7.5`, do not relabel it to `0.5` merely to make it load: stop and contact the
project owner. Historical artifacts may be preserved as evidence, but they are
not inputs to the new Torus training path.

## What the launcher does

`./dev-gocube`:

1. resolves the repository root, so it works regardless of the shell's current directory;
2. uses the repository's existing `.venv/bin/python` directly, so manual virtualenv activation is not required;
3. verifies that the local `checkpoint/` directory exists;
4. does **not** assign semantics to ambiguous pre-manifest legacy checkpoints automatically;
5. leaves compatible existing manifests unchanged;
6. fails clearly if `127.0.0.1:8765` is already occupied instead of silently starting another service;
7. runs the existing Protocol V1 service on `http://127.0.0.1:8765` with `device=auto`.

New training runs already write their own GoCube manifest, so they do not need to be added to a legacy bootstrap list.

Once the launcher is running, use GoCube's Development Workspace to generate and replay games. Repeated test games should be generated from the UI; `curl`, manual checkpoint IDs, JSON payloads, and repeated manifest registration are not part of the normal workflow.

Stop the service with `Ctrl+C`. Because the shell launcher uses `exec`, the service receives terminal signals directly.

## Optional overrides

The defaults match GoCube Development Workspace. Advanced local use can override the service parameters, for example:

```bash
./dev-gocube --device cpu
./dev-gocube --port 8877
```

If the port is changed, GoCube's configured AlphaZero base URL must be changed to match.
