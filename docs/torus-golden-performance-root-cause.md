# Torus Golden prepared-legality parity

The Torus neural path now consumes the shared `LegalActionContext` introduced
by PR #81. Search prepares legality once for each expanded nonterminal node and
passes that context through observation construction, neural evaluation and
root Dirichlet noise. Point transitions use `probe_action` plus a trusted child
constructor; the exact slow implementation remains available as the semantic
oracle.

The reproducible benchmark is:

```bash
.venv/bin/python tools/torus_golden_performance.py \
  --mode both \
  --output /tmp/torus-golden-performance.json
```

It uses the same deterministic model seed, 64 simulations, search seed and
early/mid/late/long-history Torus states for both modes. The reference mode
keeps the old repeated legality boundary and full-history state construction;
it is used only for comparison.

Observed CPU run on 2026-09-12:

| phase | reference search | optimized search | speedup | reference full-history validations | optimized full-history validations |
|---|---:|---:|---:|---:|---:|
| early | 0.182 s | 0.069 s | 2.63x | 3,116 | 0 |
| mid | 0.166 s | 0.068 s | 2.45x | 1,944 | 0 |
| late | 0.251 s | 0.071 s | 3.54x | 2,080 | 0 |
| long-history | 0.620 s | 0.070 s | 8.91x | 1,806 | 0 |

The optimized search counters are stable across the corpus: 64
`legal_calculations`, 1,645–1,654 exact action probes, 64 trusted child
constructions, 64 NN evaluations, zero standalone `legal_actions_calls`, zero
root-noise legality scans and one prepared root-noise reuse. Root visits remain
64 and selected actions match the reference in every state.

The same benchmark was also run against the real Stage 4 checkpoints. M0
search speedups for early/mid/late/long-history were 2.28x/2.25x/3.39x/9.72x;
M4 produced 2.29x/2.27x/3.43x/9.29x. Both checkpoints had zero optimized
full-history validations, zero optimized standalone legality calls, 64 root
visits and identical selected actions to reference mode for every phase.

The Torus parity gates are in
`tests/test_torus_golden_performance.py`: the fixed corpus compares legal
actions/masks, transitions, captures, resulting state identity and terminal
results; fixed-model PUCT compares WDL, unmasked policy, root visits, π, Q,
selected action, evaluator calls and simulation count; and deterministic game
traces compare state-before, action, π, captures and state-after for both M0
and a second fixed model snapshot.
