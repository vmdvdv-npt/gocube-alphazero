# GoCube learning sanity (2026-09-09)

This document records the small, reproducible post-fix diagnostic run through
the official pinned training path. It is a pipeline check, not a production
training claim.

## Current pinned run

The run used Cube4 Japanese, the baseline diversified pinned game, `komi=0.5`,
five iterations of four games, one self-play simulation, no fast games,
`train_samples_per_new_sample=1`, batch size 32, and Arena disabled during
training. It completed 20 games, saved 2,855 replay rows, and performed 93
optimizer updates over 2,855 examples. Per-iteration replay rows were
673, 522, 580, 516, and 564.

The run produced checkpoints `iteration-0000` through `iteration-0005` with
the pinned contract:

```text
searchContractId=katago-pinned-search-v4
observationSchema=gocube-observation-v4-pass-would-end-phase
valueTarget=win-loss-noresult-s1-v2
komi=0.5
```

The fixed, noise-free, unbatched Arena evaluator then compared the checkpoints
with exactly 50 simulations per move:

```text
iteration-0005 vs iteration-0002: 8W / 0L / 0D / 0NR
```

This is a positive minimal end-to-end signal. The small sample and one-sim
self-play budget are not enough to claim production playing strength; a larger
Arena is required for that.

## The earlier artifact is not a post-fix PASS

The previous text of this document claimed that a Cube3 `iteration-0005` beat
`iteration-0002` 6–2. The workspace Arena artifact with that name actually
contains 6 wins and 58 losses for the candidate, with the legacy
`gocube-observation-v3` / `gocube-search-contract-legacy` contract. It is
historical evidence of the retired path, not evidence that the current pinned
pipeline improved. The claim has therefore been removed rather than used as a
learning result.

The exact independent proofs and the hypothesis verdict table are in
`docs/GOCUBE_LEARNING_DIAGNOSTIC_20260909.md`.
