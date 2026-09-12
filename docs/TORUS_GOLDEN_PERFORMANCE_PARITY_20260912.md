# Golden Torus optimized-path performance and parity

Run: `torus-golden-stage4-seed2-parity-v8`  
Reference: `torus-golden-stage4-seed2-v4`  
Execution date: 2026-09-13 (Asia/Tbilisi)  
Base branch: `codex/torus-rebuild-v1`  
Feature branch: `codex/torus-golden-performance-parity`

## Verdict

| Gate | Result |
|---|---|
| Optimized Torus path | PASS |
| Rule semantics | IDENTICAL |
| Observation semantics | IDENTICAL |
| Evaluator semantics | IDENTICAL |
| Root-noise semantics | IDENTICAL |
| Search semantics | IDENTICAL; full-tree trace BIT-IDENTICAL |
| Full-game trace | BIT-EXACT |
| Serial/process execution | IDENTICAL |
| Training reproduction | BIT-EXACT |
| Learning system | CONFIRMED |

The active Golden contract was preserved exactly: Torus 5x5, 25 points and 26
actions including pass, komi 0.5, suicide forbidden, positional superko with
pass exempt from repetition, double-pass termination, graph-area scoring, and
no resign/cleanup/Benson behavior.

## Correctness evidence

- The fixed rules corpus covered 184 states, 5,888 action cases, 2,970 legal
  transitions, capture/suicide/superko/pass/terminal categories, and six
  deterministic fuzz trajectories (168 state/action combinations); mismatches: 0.
- Observation and evaluator checks covered 173 states. Masks, state keys,
  normalized policies and WDL outputs were bit-identical; maximum absolute
  differences were 0.0.
- Root Dirichlet-noise checks covered 23 states, including repeated same-seed
  draws and RNG-consumption checks; mismatches: 0.
- Search checks covered 23 states at 64 simulations with `cpuct=1.25` and
  `fpu=0.0`; root visits, selected actions and full-tree traces were identical.
- Four deterministic full-game traces (M0 and M4) were bit-exact, including
  state-before, action, policy, captures, state-after, score and `z` target.
- Serial/process checks were identical for the fixed four-game corpus.

The optimized hot path uses one prepared legality context per expanded node,
exact superko membership lookup, `probe_action` plus trusted internal child
construction, prepared observation/evaluation, and prepared root-noise reuse.
The audit found no full-history validation or direct standalone legality scan
in the optimized search path.

## Performance

The deterministic CPU benchmark used M0, 64 simulations, search seed 123 and
the same early/mid/late/long-history states for both modes. Reference means the
old repeated-legality/full-history path; optimized means the new prepared path.

| Phase | History | Reference search | Optimized search | Speedup | Optimized full-history validations |
|---|---:|---:|---:|---:|---:|
| early | 1 | 0.141286 s | 0.063660 s | 2.219x | 0 |
| mid | 8 | 0.138098 s | 0.061652 s | 2.240x | 0 |
| late | 24 | 0.183256 s | 0.054523 s | 3.361x | 0 |
| long-history | 120 | 0.449488 s | 0.049736 s | 9.037x | 0 |

The optimized search used zero standalone `legal_actions` calls, 64 trusted
child constructions, one prepared root-noise legality reuse, and only 49–64
legality calculations for 64 simulations, depending on the state.

## Stage4 reproduction

The canonical run used the frozen Stage4 schedule and replay seed namespace;
no new parameter experiment or ablation was introduced.

- Self-play: **512/512 valid games**, **15,455 positions**, **0 technical games**.
- Replay chunks were exact: 4,025 / 3,316 / 4,149 / 3,965 rows, respectively.
- New replay and self-play semantics matched the reference lineage exactly.
- M0–M4 model hashes matched the immutable reference byte-for-byte at the
  model level; only run metadata fields were allowed to differ.

| Checkpoint | Model hash |
|---|---|
| M0 | `sha256:64c6d899ad85d056340c811db1eded7bf497812f9d5205dd50fb84961486976e` |
| M1 | `sha256:bcc378cc355979192ae272515f1a0f9e8de2e01105c5402be27ad105c5290d44` |
| M2 | `sha256:374bcd5a6fdc270ac3863f573970e12fc77f08e9e00bec0c6ee4097b275eaedf` |
| M3 | `sha256:b2a5619b04708c492c1be44c01fdae6c1c816ecc7c9f45b98763a2e83b85e854` |
| M4 | `sha256:b208a0b0464288950c7f3ce0c9ac64191274532661184e7ad1bc7573699b850d` |

## Arena evidence

All Arena runs used 64 start-pairs, 128 games for primary comparisons, 16
start-pairs/32 games for the frozen adjacent-progression subset, temperature
0, no noise, no resign, 64 simulations, `cpuct=1.25`, and technical games 0.
Intervals are the bounded independent start-pair mean 95% Hoeffding intervals.

| Comparison | Games | W/L/D | Mean pair score | 95% interval | Technical |
|---|---:|---:|---:|---:|---:|
| M1 vs M0 | 128 | 91/37/0 | 0.7109 | [0.5412, 0.8807] | 0 |
| M4 vs M0 | 128 | 124/4/0 | 0.9688 | [0.7990, 1.0000] | 0 |
| M4 vs M1 | 128 | 113/15/0 | 0.8828 | [0.7130, 1.0000] | 0 |
| M2 vs M1 | 32 | 25/7/0 | 0.7813 | [0.4417, 1.0000] | 0 |
| M3 vs M2 | 32 | 19/13/0 | 0.5938 | [0.2542, 0.9333] | 0 |
| M4 vs M3 | 32 | 18/14/0 | 0.5625 | [0.2230, 0.9020] | 0 |

## Artifacts and reproducibility

- Machine-readable report:
  `runs/torus-golden-stage4/torus-golden-stage4-seed2-parity-v8/final-report.json`
- Run manifest:
  `runs/torus-golden-stage4/torus-golden-stage4-seed2-parity-v8/manifest.json`
- Source commit used for the canonical run:
  `958c26fc565fcef4bd1d693cc3b8b1b8524c8f5e`
- Source tree used for the canonical run:
  `5f9f24e8301f7315e458f1f6d2ece2e1f24a3099`
- Immutable reference artifact root:
  `runs/torus-golden-stage4/torus-golden-stage4-seed2-v4/`

The canonical parity command was:

```bash
.venv/bin/python tools/torus_golden_parity.py \
  --run-id torus-golden-stage4-seed2-parity-v8 \
  --workers 16
```

## Regression suite

The full local suite completed in 368.67 seconds with **1,197 passed, 1
failed, 57 warnings**. The sole failure was the pre-existing data-dependent
assertion in `tests/test_gocube_s1_replay_audit.py`: the ignored local
`data/` directory contains 8,208 fork records while that test expects zero.
No tracked test or user data was changed. The Torus targeted regression set
passed (62 tests before the full-suite run).

