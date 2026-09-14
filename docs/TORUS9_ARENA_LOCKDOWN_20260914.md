# Torus 9×9 Arena lockdown — superseded

The temporary Torus9-specific lockdown from PR #96 is superseded by the
repository-wide single Arena engine documented in:

`docs/ARENA_SINGLE_ENGINE_20260914.md`

Current production entry point:

```bash
.venv/bin/python tools/arena.py --candidate PATH.pt
```

Torus 9×9 is now a profile of that engine, not a separate Arena
implementation.

Historical Arena executors remain frozen and require the explicit
`--allow-frozen-arena` reproduction override. They are not current Golden
production paths.
