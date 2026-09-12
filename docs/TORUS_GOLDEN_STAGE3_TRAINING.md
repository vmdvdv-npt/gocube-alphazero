# Stage 3 — Golden Torus neural training proof

Stage 3 introduces a separate training identity:
`gocube-torus-golden-training-v1`, defined by
[`configs/gocube/torus_golden_training_v1.json`](../configs/gocube/torus_golden_training_v1.json).
The profile references, but does not rewrite, the frozen Golden v2 rules,
topology, observation, target, and Arena identities.

The neural decision is **NEW MINIMAL GOLDEN IMPLEMENTATION**. The existing
graph network was inspected, but it is coupled to the legacy game wrapper and
optional legacy heads. `GoldenGraphNetV1` is therefore a pure PyTorch graph
network with 6 input channels, four residual message-passing blocks, a 26-way
policy head, and a 3-way side-to-move `[WIN,DRAW,LOSS]` head. Adjacency is
copied from `GoldenTopology` at construction; no manually duplicated board
table is used.

Self-play and Arena are separate contracts. Self-play uses root-only
Dirichlet noise and temperature-1 visit sampling for plies 1–8. Arena remains
the frozen noise-free sequential 64-simulation Golden PUCT. Technical
self-play games are retained as evidence but are rejected by replay and fail a
canonical chunk.

The canonical runner is:

```bash
.venv/bin/python tools/torus_golden_stage3_train.py \
  --run-id torus-golden-stage3-seed1 \
  --profile proof-standard \
  --device auto \
  --workers 16
```

Before chunk 1, the runner performs a serial-vs-parallel equivalence gate for
fixed game IDs and seeds. The allowed performance change is independent-game
parallelism only: one immutable model instance, inference batch size 1,
coalescing disabled, and sequential PUCT inside each game. Replay rows are
sorted by `(game_id, ply)` after the chunk completes. No batched inference,
parallel MCTS, fast search, or worker-derived randomness is enabled.

Canonical artifacts are written beneath `runs/torus-golden-stage3/<run_id>/`
and are intentionally ignored by Git. The final report records checkpoint
lineage, parameter/model/artifact hashes, self-play provenance, optimizer
metrics, pre-generated Arena starts, Arena checkpoint loads, and one of the
three Stage-3 verdicts.

