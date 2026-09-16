# Raw Arena Legion sweep artifacts

These directories are the raw outputs of the final fixed-seed Arena sweep on
2026-09-16. Any Stage 7 baseline referenced by these artifacts is historical /
retired evidence; the Stage 7 harness is not a current entrypoint. Full runs contain `summary.json`, `manifest.json`, `games.jsonl`,
and `hardware-telemetry.jsonl`; early-gate diagnostics also contain
`performance-degraded.json`.

| directory | role |
|---|---|
| `arena-sweep-a-c1-w6-g64-strict` | Stage A 16-context early degraded diagnostic |
| `arena-sweep-a-c2-w6-g64-strict` | Stage A 32-context early degraded diagnostic |
| `arena-sweep-a-c4-w6-g64-strict` | Stage A 64-context run |
| `arena-sweep-a-c6-w6-g96-strict` | Stage A 96-context run |
| `arena-sweep-a-c8-w6-g128-strict` | Stage A 128-context run |
| `arena-sweep-a-c12-w6-g192-strict` | Stage A 192-context run |
| `arena-sweep-b-c12-w4-g192-strict` | Stage B wait4 finalist |
| `arena-sweep-b-c12-w8-g192-strict` | Stage B wait8 control |
| `arena-sweep-b-c12-w12-g192-strict` | Stage B wait12 control |
| `arena-sweep-b-neighbor-c8-w4-g128-strict` | Stage B neighboring 128-context wait4 control |
| `arena-sweep-b-neighbor-c8-w8-g128-strict` | Stage B neighboring 128-context wait8 control |
| `arena-sweep-b-neighbor-c8-w12-g128-strict` | Stage B neighboring 128-context wait12 control |
| `arena-sweep-c-c12-w4-cap16-g192-strict` | Stage C cap16 early degraded diagnostic |
| `arena-sweep-c-c12-w4-cap32-g192-strict` | Stage C cap32 control |
| `arena-sweep-d-w8-c8-w4-g64-debug` | Stage D 8-worker debug diagnostic |
| `arena-sweep-d-w12-c5-w4-g64-debug` | Stage D 12-worker debug diagnostic |
| `arena-sweep-repeat-top-c12-w4-g192-strict` | finalist repeat |
| `arena-final-confirm-c12-w4-g256-strict` | final long confirmation |

The manifests preserve the original run provenance, including the Legion
checkpoint paths and master seed. The report's comparative values are read
from these raw `summary.json` files rather than copied by hand.
