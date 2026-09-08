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
