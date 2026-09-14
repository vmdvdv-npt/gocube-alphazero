# Torus 9×9 Arena lockdown — superseded

The temporary fail-closed lockdown from PR #96 has been superseded by the
single production Arena implementation documented in
`docs/TORUS9_ARENA_SINGLE_PRODUCTION_20260914.md`.

Current production entry point:

```bash
.venv/bin/python tools/torus9_arena.py --candidate PATH.pt
```

Historical Arena executors remain frozen and require the explicit
`--allow-frozen-arena` reproduction override. They are not current Golden
production paths.
