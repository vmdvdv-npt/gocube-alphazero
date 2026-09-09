# GoCube V3 learning sanity (2026-09-09)

This is a post-fix diagnostic run through the official pinned training path,
not through `alphazero.envs.gocube.train`.

Configuration:

- topology: Cube4 Japanese V3;
- `komi=0.5`;
- 5 iterations, 4 games per iteration;
- pinned KataGo search, 1 self-play simulation, fast-game probability 0%;
- replay training at 1 configured train sample per new sample;
- Arena/model gating disabled for training, as required for this sanity run.

Training completed all five iterations:

| Metric | Result |
| --- | ---: |
| Games | 20 |
| New replay samples | 2,088 |
| Optimizer steps | 11 planned/11 actual |
| Iteration-5 samples | 350 |
| Iteration-5 optimizer steps | 2 planned/2 actual |
| Fast-game fraction | 0% |

The official unbatched fixed-50-simulation checkpoint Arena then compared
iteration 5 against iteration 2 for 8 games: iteration 5 won 6, lost 2, and
drew 0 (`75.0%`, Wilson 95% interval `[40.9%, 92.9%]`). This small Arena is a
pipeline sanity check, not evidence of production-strength playing quality.

The checkpoint manifest for this fresh run records
`searchContractId=katago-pinned-search-v4`, the pinned reference commit, V3
target schemas, and `komi=0.5`. The earlier repository-local 15–113-related
artifacts are retained as historical evidence; their manifest records
`searchContractId=gocube-search-contract-legacy`, so they were not a valid
post-fix comparison and are exactly the unsafe path this change retires.
