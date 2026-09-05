# C4-T001 / C4-C001

Status: `INTERRUPTED_BY_OPERATOR`

Reason: Arena after iteration 1 was intentionally stopped because per-iteration Arena was found to be unnecessarily expensive for this baseline.

Stop date: 2026-09-05 (Asia/Tbilisi; SIGINT sent after the pre-stop snapshot)

Run name: `c4-t001-c4-c001`

Source git commit: `85c87a7cfd467a4d3f4b2844253fb63d746d672a`

Launch command:

```bash
PYTHONPATH=. .venv/bin/python alphazero/envs/gocube/train.py \
  --topology cube \
  --size 4 \
  --workers 2 \
  --sims 100 \
  --arena-sims 100 \
  --games-per-iteration 256 \
  --iterations 5 \
  --train-batch-size 256 \
  --fast-game-prob 0.25 \
  --endgame-sample-weight 1 \
  --run-name c4-t001-c4-c001
```

Effective parameters: topology `cube`, size `4`, workers `2`, regular MCTS `100` sims, fast MCTS `20` sims with probability `0.25`, `256` self-play games per iteration, training batch size `256`, endgame sample weight `1`, Arena `100` sims and `128` Arena games per iteration.

Iteration 1:

- Self-play completed: `256/256` games.
- Game records saved: `256`.
- Game ID range: `C4-000263` through `C4-000518`.
- Training completed and `iteration-0001.pkl` was saved.
- Arena was partial: last console progress before SIGINT was `51/128`, winrates `[0.980, 0.020]`, no-result `0`.
- Iterations `2`–`5` were not started.

Saved artifacts:

- `checkpoint/c4-t001-c4-c001/iteration-0000.pkl`
- `checkpoint/c4-t001-c4-c001/iteration-0001.pkl`
- `checkpoint/c4-t001-c4-c001/gocube-run.json`
- `data/c4-t001-c4-c001/iteration-0001-data.pkl`
- `data/c4-t001-c4-c001/iteration-0001-policy.pkl`
- `data/c4-t001-c4-c001/iteration-0001-value.pkl`
- `data/c4-t001-c4-c001/iteration-0001-ownership.pkl`
- `data/c4-t001-c4-c001/iteration-0001-ownership-mask.pkl`
- `data/c4-t001-c4-c001/iteration-0001-score.pkl`
- `data/c4-t001-c4-c001/records/iteration-0001/` (256 records)
- `data/c4-t001-c4-c001/records/iteration-0001/iteration-manifest.json`
- `runs/c4-t001-c4-c001/events.out.tfevents.1788633893.Legion`

SHA256:

- `iteration-0001.pkl`: `64cf800460f6090880c7818cbeff80123257dcd14c79689f108cc5523fb58722`
- `iteration-manifest.json`: `909252d5d793c446b163837019098cb60036a5fa6c18b390ee154bcb9ff3414a`

The checkpoint was verified readable with `torch.load`. The iteration-1 data and all 256 records remain available for analysis. The Arena iteration is incomplete; no Arena completion result was recorded. Iterations 2–5 and no training code were changed. The global C4 Game ID counter was not reset or modified by this operation; it remains at `next_number: 519`.

No console log file existed in the repository; the Arena progress above was captured from the live PTY immediately before SIGINT.
