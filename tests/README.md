# Test layout

Pytest discovers tests recursively under `tests/`.

- `unit/` — deterministic behavior of one domain or helper.
- `integration/` — HTTP, multiprocessing, and cross-component execution.
- `contracts/` — frozen Golden boundaries, protocol rules, parity, and repository policy.
- `reference/` — independent topology checks and pinned KataGo differential tests;
  `reference/katago/` contains their fixtures and test modules.
- `fixtures/` — versioned test data; `support/` — reusable test-only helpers.
- Benchmark plans live in `docs/benchmarks/`; generated run data is kept outside
  the tracked test tree.

Useful commands:

```bash
.venv/bin/python -m pytest -q -m "not katago_reference"
.venv/bin/python -m pytest -q -m katago_reference
```

The KataGo command requires the pinned local oracle. Routine cleanup may remove
`__pycache__/` and `.pytest_cache/`; do not remove `runs/`, `data/`, checkpoints,
or experiment archives without an explicit retention decision.
