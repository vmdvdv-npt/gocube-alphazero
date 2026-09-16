# Torus 9×9 current Golden M1 baseline review

- Run: `torus9-golden-v3-20260914-run03`
- Base commit: `53946d0c84fca5a6f81a387bfd399ea62e34b088`
- Run git SHA: `d54de530fc4026d22409ca017f329f88a920dec2`
- Device: `cuda` (locked; RTX 3060)
- Resolved config fingerprint: `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`
- Golden Standart: read-only; unchanged

## M0→M1

| Metric | Value |
|---|---:|
| M1 checkpoint | `runs/torus9/active/torus9-golden-v3-20260914-run03/checkpoints/M1.pt` |
| Wall time | 2260.3593 s |
| Games / technical games | 64 / 0 |
| Total moves / moves per second | 4993 / 2.20894 |
| Inference calls / rows / rows per second | 315477 / 315477 / 139.5694 |
| Batch rows mean / p50 / p95 | 1 / 1 / 1 |
| GPU utilization average / peak | 34.8683% / 79% |
| VRAM peak | 1097.6758 MiB |
| Max active MCTS lanes | 16 |

The baseline used `coalescing=OFF`, so each inference request was one row.
GPU values are from 1784 persisted NVML samples; wall time is measured from
the M0 checkpoint timestamp to the completed self-play artifact timestamp.

## Stop and resume contract

- Status: `STOPPED_AFTER_M1`
- Result: `PAUSED_FOR_BASELINE_TELEMETRY_REVIEW`
- Reason: `manual checkpoint for baseline telemetry review`
- Next transition started: `NO`
- Replay after generation 1: unchanged (`sha256:38c190f1f69988928475f6ba841fe63e04f0c4683b1bfe3695891f354cdd9214`)
- Optimizer state: present; Adam step `80`
- Resume point: `M1.pt` + `replay/rolling-after-01.jsonl`; explicit resume required
- `M2`/iteration-2/benchmark artifacts: absent
