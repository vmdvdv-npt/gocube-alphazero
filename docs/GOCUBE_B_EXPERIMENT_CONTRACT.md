# GoCube B0/B1 experiment contract

The hardened B entrypoint is:

```bash
.venv/bin/python tools/gocube_b_experiment.py --treatment B0 --heldout-suite path/to/frozen-suite.json
.venv/bin/python tools/gocube_b_experiment.py --treatment B1 --heldout-suite path/to/frozen-suite.json
```

`--treatment` is the only model choice exposed to the operator. The launcher
maps `B0` to `baseline` and `B1` to `g1`, pins the Cube-4 production settings,
and runs the B0 and B1 effective-config builders in separate Python
processes. A run cannot start if the configs differ outside the structural
model/profile and run-identity whitelist. The default scientific stopping
target is 40,000,000 cumulative newly accepted replay samples; it can be
changed explicitly to an optimizer-example target:

```bash
.venv/bin/python tools/gocube_b_experiment.py \
  --treatment B1 --heldout-suite path/to/frozen-suite.json \
  --cumulative-optimizer-examples-target 40000000
```

`--heldout-suite` is mandatory for a real run. Its SHA-256 is computed from
the bytes of that frozen artifact and is stored in the immutable contract. If
the suite is not available yet, the launcher permits only `--dry-run`; it does
not write a runnable contract record and cannot start training.

The seed protocol is immutable: `seed_list=(0,1,2,3,4)`, with seeds `0, 1,
2` mandatory for both treatments. The initial analysis therefore uses 3 B0
and 3 B1 runs. Seeds `3` and `4` are extension seeds, permitted only with a
machine-readable approval from the pre-registered ambiguity/variance
criterion. The contract records `initial_seed_count=3` and
`extension_seed_count=5`, so the same v1 contract covers both 3+3 and 5+5
analysis without silently changing the seed universe.

For an extension seed, provide `--extension-seed-decision decision.json` with
the contract criterion ID and machine-readable `criterion_evidence` showing
either ambiguity or excess variance. The launcher also binds that approval to
the generated contract SHA before training. The decision is valid only at the
100% milestone recorded by that same immutable contract: this is
`cumulative_new_samples=40,000,000` for the default contract, or the final
optimizer-example target for an explicitly selected optimizer-example
contract.
The B launcher never accepts `--allow-dirty-source`; B runs require a clean
committed source tree. The contract SHA-256 is passed to the child trainer and
stored in its run manifest and checkpoints.

The canonical training batch `1024` is not an experiment marker. Ordinary
Cube-4 production training may use that batch without a B contract. A B run is
activated only when the launcher passes both the explicit
`gocube-b-experiment-contract-v1` marker and the generated contract SHA-256;
missing or mismatched marker/SHA metadata fails closed.

`--iterations` is only a safety ceiling. `--games-per-iteration=256` remains
the generation chunk and is not the experimental budget. The production loop
generates a chunk, counts the actual accepted rows, trains according to the
configured samples-per-new-sample ratio, records any final-chunk overshoot,
and repeats until the cumulative target is reached.

Before the child training process starts, the launcher writes one
machine-readable `gocube-b-experiment-contract.json` containing:

- the immutable `gocube-b-experiment-contract-v1` specification;
- source SHA, rules/search/termination/target identities, both model
  contracts, budgets, optimizer/scheduler, seeds, evaluation semantics, and
  the SHA-256 of the actual frozen held-out-suite artifact;
- both effective configs, their diff, and the allowed semantic difference
  paths.

Every training namespace publishes `data/<run>/training-progress.json`. Its
canonical cumulative counters are self-play games completed, positions
generated, saved replay samples, newly accepted samples, optimizer steps, and
optimizer examples seen. The same counters are recorded per iteration and in
explicit `cumulative_*` form. Replay-window rows are sampling history only and
are never counted as newly accepted samples. Episode move-limit terminations,
`NO_RESULT`, and average game length are reported separately because they can
change samples per game.

The evaluation result semantics are fixed as W=1, D=0.5, NR=0.5, L=0 on
paired starting positions. The primary endpoint is the paired position score;
uncertainty is identified as hierarchical paired bootstrap over seeds and then
starting-position pairs. `NO_RESULT` remains in the denominator and contributes
0.5. Reports for both B0 and B1 include games, positions, saved/new samples,
optimizer steps, and examples seen, so acceptance can be stated as:

> B0 and B1 were trained to the same cumulative sample budget.

## Evaluation milestones

Scientific comparisons are pinned to the cumulative sample clock, not to
iteration numbers. For the default `cumulative_new_samples` target of
40,000,000, the immutable contract records milestones at 25%, 50%, 75%, and
100%: 10,000,000; 20,000,000; 30,000,000; and 40,000,000 accepted samples.
For an explicitly selected optimizer-example target, the same immutable
contract records the corresponding 25%, 50%, 75%, and 100% optimizer-example
milestones. B4 evaluation and aggregation require that contract record and
reject clocks or milestones not present in its schedule. Bootstrap iteration,
health-reference iteration, arena-anchor period, and the held-out suite size
remain fixed evaluation metadata, not scientific clocks.

## Canonical Cube-4 training batch

The approved canonical hardened Cube-4 training batch is **`1024`**.

This value is intentional and supersedes the older launcher value of `256`.
For the B0/B1 experiment, `1024` is the source-of-truth value and must agree
across the shell launcher, `Cube4ProductionContract`, the B experiment
contract, checkpoint validation, and both effective configs.

A future reduction to `512` is permitted only by a new explicit project
decision accompanied by a contract/documentation update. Values below `512`
are not approved for this experiment. Code or documentation that still treats
`256` as the canonical production training batch is stale and must not be used
to override the B experiment contract.

## Frozen evaluation suite and B4 statistics

The canonical model-independent held-out suite is committed at
`evaluation/gocube-b-heldout-suite-v1.json`. Its identity is
`gocube-b-heldout-suite-v1`, generated by
`gocube-b-heldout-generator-v1` with master seed `20260908`, and its exact
file SHA-256 is
`0a0d6b9662532aa1bd6fcfad2662651adfc9fc4f6058082a0ad22f86ab319e9d`.
It contains exactly 16 Cube-4 Japanese positions: four each after 12, 24,
36, and 48 legal point moves. The generator uses an independent
`random.Random(20260908 + position_index)` stream, authoritative legal moves,
canonical PointId order, and no model, self-play, replay, color swap, or
symmetry input. `tools/build_gocube_b_heldout_suite.py --check` must reproduce
the committed bytes and replay every timeline/fingerprint.

Both `tools/evaluate_gocube_b_experiment.py` and
`tools/analyze_gocube_b_evaluation.py` require the immutable B contract record
via `--experiment-contract`. They derive the scientific clock and allowed
milestones from that record, so an optimizer-example contract is evaluated on
its own 25/50/75/100% schedule.

`tools/evaluate_gocube_b_experiment.py` is the only production evaluator for
B checkpoints. It selects the first committed/resumable checkpoint whose
cumulative scientific counter is at least the requested milestone, and records
iteration, checkpoint SHA-256, cumulative counters, target, and overshoot for
both profiles. Each frozen starting position is one statistical unit and runs
exactly two games: B0 black/B1 white and B1 black/B0 white. Both games replay
the same semantic starting state; the stones are never color-inverted. Search
is fixed at 50 simulations, fast simulation probability 0, root noise and
root policy temperature off, and action/arena temperature 0. B0 receives its
18-channel adapter and B1 its 20-channel adapter from the existing checkpoint
contract.

The game score is W=1, D=0.5, `NO_RESULT`=0.5, L=0. A pair score is the mean
of its two games, so `NO_RESULT` is retained in the denominator. Seed score is
the mean of 16 pair scores; overall score gives every training seed equal
weight. `tools/analyze_gocube_b_evaluation.py` uses statistical method
`hierarchical-paired-bootstrap-seeds-to-starting-position-pairs-v1`: 10,000
deterministic NumPy-generator replicates with seed `20260908`, resampling
training seeds first and 16 position pairs within each selected seed. The
reported 95% percentile interval is on overall delta from 0.5. Classification
is `B1_BETTER` when its low endpoint is positive, `B0_BETTER` when its high
endpoint is negative, and `INCONCLUSIVE` otherwise.

After mandatory seeds 0, 1, and 2 have been evaluated at the 100% milestone
of their immutable B contract (the default is
`cumulative_new_samples=40,000,000`), extension to seeds 3 and 4 is permitted
only when the pre-registered criterion
`extend-to-five-seeds-only-if-mandatory-seed-bootstrap-ambiguity-or-variance-v1`
finds bootstrap ambiguity or sample standard deviation of mandatory seed
deltas at least 0.10. The decision JSON must carry the contract's
`scientific_clock`, its final `scientific_milestone`, and the contract SHA;
decisions cannot be created or used at an intermediate milestone or at an
unregistered clock or target. The extension decision is written separately as
JSON.
The legacy `tools/evaluate_gocube_checkpoints.py` aggregate Wilson evaluator
rejects checkpoints marked with the B contract and cannot be used for final B
analysis.
