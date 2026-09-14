# Current Golden boundary — Torus 9×9 M0→M17 (2026-09-15)

This document freezes the production boundary proven by the successful Torus 9×9 line `M0→M17`. It is an audit/ownership document, not a redesign of self-play, training, MCTS, the network, or Arena.

## 1. Immutable reference

| Item | Immutable value |
|---|---|
| Reference run | `torus9-golden-v3-20260914-run03` |
| PR #93 merge commit | `152be924db12530ff04b37266925838da191e948` |
| PR #93 head | `2e70a3a3e6a64e82fa784d6760df9c51283b2e65` |
| Common Git tree | `d4b438ff6b8442aae03ef16cafacbad4ad7ece14` |
| M1 baseline run Git SHA | `d54de530fc4026d22409ca017f329f88a920dec2` |
| Original base commit | `53946d0c84fca5a6f81a387bfd399ea62e34b088` |
| Reference profile fingerprint | `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775` |
| Last clean resumable boundary | M17 |
| M18 | never started |

The run report is `docs/TORUS9_GOLDEN_CURRENT_V3_M1_M17_NIGHT_RUN_20260914.md`. The external run namespace is `runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03` on Legion. M2…M17 completed with 64 games/iteration and `technical=0`.

The reference profile fingerprint above remains an immutable provenance identifier for the historical run. This PR removes the forbidden historical komi sentinel metadata from the current profile. Because the profile fingerprint hashes the whole JSON payload, the sanitized current content fingerprint is therefore `sha256:d3620fc36600d36753a4bb51a9810b7ea21fe82a43f70a43b6363485dc9684e3`. No topology, rules, observation, target, network, search, replay, training, seed, checkpoint, or Arena semantics changed.

## 2. Official production entrypoints

The supported current boundary is intentionally small:

| Purpose | Entrypoint / boundary | Status |
|---|---|---|
| Continue/resume current Torus9 Golden training | `tools/continue_torus9_golden_m1_m100.py` | KEEP CURRENT |
| Pinned continuation runtime loaded by launcher | `tools/_frozen_continue_torus9_golden_m1_m100.py` | KEEP CURRENT |
| Standalone Arena CLI | `tools/arena.py` | KEEP CURRENT |
| Standalone Arena engine | `tools/arena_engine.py` | KEEP CURRENT |
| Torus9 Arena adapter/profile | `tools/arena_profiles/torus9.py` | KEEP CURRENT |
| Torus9 checkpoint/replay/training library boundary | `gocube_golden/torus9.py` | KEEP CURRENT |
| GoCube product Protocol V1 HTTP service | `python -m alphazero.envs.gocube.integration.server` | KEEP CURRENT |

No separate current checkpoint-inspection CLI is claimed by this audit. Checkpoint construction/loading is a library boundary in `gocube_golden.torus9`; inventing another CLI here would widen the task without evidence.

## 3. Proven M0→M17 runtime/dependency graph

The continuation path is:

```text
tools/continue_torus9_golden_m1_m100.py
  -> importlib loads tools/_frozen_continue_torus9_golden_m1_m100.py
  -> configs/gocube/torus9_golden_current_v3.json
  -> gocube_golden.torus9_contract.load_torus9_current_profile()
  -> gocube_golden.torus9
       -> gocube_golden.topology       [topology]
       -> gocube_golden.state          [state/rules identity]
       -> gocube_golden.rules          [legal play/scoring]
       -> gocube_golden.search         [PUCT/search evaluator]
       -> gocube_golden.neural         [network/loss helpers]
       -> gocube_golden.stage3_contract[shared coefficient contract]
       -> self-play + inference coordination/batching
       -> game/replay rows
       -> replay merge/window
       -> trainer/optimizer/losses
       -> checkpoint + training state
       -> resume loaders
  -> tools.hardware_telemetry           [execution telemetry only]
```

The frozen runtime imports the current Torus9 API directly (`Torus9SelfplayConfig`, `Torus9TrainingConfig`, current network construction, self-play, replay merge, training, checkpoint load/save and Arena helpers). It does **not** route M0→M17 through `alphazero.Coach`, `alphazero.SelfPlayAgent`, `alphazero.NNetWrapper`, or the historical GoCube training launchers.

### Component ownership map

| Component | Called by / calls | Semantics vs execution | M0→M17 | Arena | GoCube product | Classification |
|---|---|---|---:|---:|---:|---|
| `configs/gocube/torus9_golden_current_v3.json` | launcher/runtime/profile validator | scientific + declared execution knobs | yes | yes | no direct | KEEP CURRENT |
| `gocube_golden/torus9_contract.py` | runtime, Arena profile, tests | scientific validation | yes | yes | no direct | KEEP CURRENT |
| `gocube_golden/torus9.py` | frozen runtime, Arena adapter | scientific + execution implementation | yes | yes | no direct | KEEP CURRENT |
| `gocube_golden/topology.py` | Torus9/rules/state | scientific | yes | yes | shared | KEEP SHARED |
| `gocube_golden/state.py` | Torus9/rules/search | scientific | yes | yes | shared | KEEP SHARED |
| `gocube_golden/rules.py` | Torus9/search | scientific | yes | yes | shared | KEEP SHARED |
| `gocube_golden/search.py` | Torus9 self-play/Arena | scientific search semantics | yes | yes | no direct | KEEP SHARED |
| `gocube_golden/neural.py` | Torus9 network/trainer | scientific model/loss semantics | yes | yes | no direct | KEEP SHARED |
| `tools.hardware_telemetry` | frozen runtime | execution only | yes | no | no | KEEP SHARED |
| `tools/arena.py` + `tools/arena_engine.py` | standalone CLI | evaluation execution | no training | yes | no | KEEP CURRENT |
| `tools/arena_profiles/torus9.py` | Arena registry/engine | current Torus9 evaluation contract | no training | yes | no | KEEP CURRENT |
| `alphazero/envs/gocube/integration/*` | GoCube frontend/API | product boundary | no | separate | yes | KEEP CURRENT |
| `alphazero/NNetWrapper.py` | GoCube integration model loader | legacy model adapter | no | no current Arena | yes today | KEEP TEMPORARILY FOR MIGRATION |
| `alphazero/GenericPlayers.py` + `alphazero/MCTS.pyx` | GoCube game generator | legacy player/search adapter | no | no current Arena | yes today | KEEP TEMPORARILY FOR MIGRATION |

## 4. Scientific semantics vs execution mechanics

### Scientific contract — frozen to M0→M17

**Game/rules**

- topology: Torus 9×9, 81 points, row-major point IDs, wrap X/Y;
- scoring: exact graph-area;
- ko: positional superko;
- suicide: forbidden;
- termination: two consecutive passes;
- Benson automatic ending: OFF;
- komi: **0.5 only**.

**Observation/targets**

- observation: `6×81`, channels `own_stones`, `opponent_stones`, `side_to_move_color`, `previous_pass`, `legal_point_mask`, `komi`;
- action count: 82; pass index: 81;
- observation fingerprint: `sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f`;
- target: W/D/L from side-to-move perspective;
- policy target: root visit distribution before chosen move;
- ownership and score auxiliary targets: ON;
- technical outcomes: excluded;
- target fingerprint: `sha256:02ab244688534b271473302ab4edf00516b91d43fb91b8c9e592d2a8de63dfb5`.

**Network**

- architecture: `GoldenGraphNetV2-Torus9`;
- hidden/channels: 80;
- blocks: 8;
- input channels: 6;
- heads: policy `[82]`, value `[3]`, ownership `[81,3]`, score `[1]`;
- explicit symmetry augmentation: OFF.

**Self-play/search**

- 64 games/iteration; 64 simulations;
- cpuct 1.25; FPU 0.0;
- root noise ON; epsilon 0.25; alpha 0.11;
- temperature 1.0 on plies 1–8, then 0;
- fast search OFF; resign OFF; watchdog 500;
- self-play contract fingerprint: `sha256:22a4e4dd37d70bd3d712b909476120b96385802ec874b99358fab256c4e3351f`;
- master self-play seed `202609131002` with deterministic seed derivation in provenance helpers.

**Replay/training**

- replay: rolling last 3 generations, cap 20,000, deterministic/reproducible sampling;
- optimizer: Adam; LR 0.001; weight decay 0;
- batch size 64; 80 optimizer steps/iteration; 5,120 samples consumed/iteration;
- LR scheduler: none; model gating: false;
- ownership loss ON; score loss ON;
- model-init/training/Arena/evaluation seeds: `202609131001/1003/1004/1005` respectively.

**Checkpoint/resume**

The current frozen runtime uses the Torus9 checkpoint and training-state load/save functions in `gocube_golden.torus9`, and the M17 run report records M17 as the last clean resumable boundary with `continuation-state.json`, `iter-17-summary.json`, `manifest.json`, checkpoints and replay retained in the Legion run namespace. These large artifacts are not committed to GitHub, so GitHub CI cannot honestly re-open the physical M17 weights. This repository audit protects the loader/profile/resume contract and records the external artifact path; physical M17 load remains an artifact-location check on Legion rather than a fake repository unit test.

### Execution mechanics — explicitly not scientific selection

The current profile records workers, coalescing/batch-cap/wait candidates, batching and telemetry separately. The M1→M17 report shows that execution was performance-degraded despite scientific validity: observed process CPU was about 1.11 core-equivalent with `workers=16`; M4…M17 mean inference batches were mostly about 9–10 rows and below the desired 16-row threshold; sampled GPU average was roughly 26–33%. The measured historical 12/4 cap/wait choice is telemetry only and is not a new Golden semantic standard.

No process/thread architecture or throughput optimization is changed in this PR.

## 5. Canonical source of truth

The only authoritative current Torus9 profile is:

`configs/gocube/torus9_golden_current_v3.json`

Parameter ownership is:

| Parameter family | Canonical source | Consumers | Duplicates / policy |
|---|---|---|---|
| topology/rules/komi | current profile + strict `torus9_contract` validation | frozen runtime, Torus9 API, Arena adapter | older profiles are historical only; never auto-selected |
| observation/target fingerprints | current profile + contract constants | network/self-play/training/tests | validated fail-closed |
| network 80×8/heads | current profile + contract validation | network builder/checkpoint loader | historical 64×8 profile remains legacy |
| self-play semantics | current profile | frozen runtime/Torus9 API | legacy profile values cannot be launcher defaults |
| replay/training budget | current profile | frozen runtime/Torus9 trainer | no CLI default may silently replace it |
| seeds | current profile | frozen runtime/provenance | validated positive and exact |
| Arena Torus9 settings | current profile consumed by `tools/arena_profiles/torus9.py` | standalone Arena | no training/gating coupling |

The current launcher is guarded against explicit references to retired profiles and legacy training symbols. The protective tests also require exactly one `status: current` Torus9 JSON profile.

## 6. Legacy inventory

### KEEP CURRENT

- `tools/continue_torus9_golden_m1_m100.py`;
- `tools/_frozen_continue_torus9_golden_m1_m100.py`;
- `configs/gocube/torus9_golden_current_v3.json`;
- `gocube_golden/torus9.py` and `gocube_golden/torus9_contract.py`;
- standalone Arena CLI/engine/registry/Torus9 adapter;
- `alphazero/envs/gocube/integration/*` as the Protocol V1 product surface.

### KEEP SHARED

- `gocube_golden/{topology,state,rules,search,neural,provenance,arena_contract}.py` and directly used shared contract helpers;
- telemetry helpers used by the frozen runtime.

### KEEP TEMPORARILY FOR MIGRATION

- `alphazero/NNetWrapper.py`: imported directly by `integration/models.py` to load historical `.pkl` checkpoints;
- `alphazero/GenericPlayers.py` and `alphazero/MCTS.pyx`: the Protocol V1 `GameGenerator` currently creates `MCTSPlayer` from this stack;
- historical GoCube checkpoint/catalog metadata compatibility needed by the current Protocol V1 loader until a current Golden model adapter replaces it.

### LEGACY — REMOVE IN LATER STAGE

- `alphazero/Coach.py`;
- `alphazero/SelfPlayAgent.pyx` for the current Golden line (still compiled by broad legacy CI and therefore should be removed together with that legacy contract, not piecemeal);
- old Torus9 learning/ownership experimental profiles, including `torus9_golden_learning_v1.json` and `torus9_ownership_ab_v1.json`;
- old Torus/Golden stage profiles and frozen experiment configs not selected by the current launcher;
- old Torus training/demo/calibration/ownership-A/B entrypoints;
- old checkpoint-Arena/gating runners superseded by the standalone Arena engine;
- historical one-off diagnostics and documentation that describe those paths as current.

### UNKNOWN — INVESTIGATE before deletion

Any historical loader/migration fixture that is still referenced by integration tests but whose production requirement is not demonstrated must remain UNKNOWN rather than being promoted to current. The next cleanup PR should use import/test evidence to resolve these one by one.

## 7. GoCube product boundary

Protocol V1 is a stable frontend-facing boundary implemented by `alphazero/envs/gocube/integration/server.py` and `service.py`:

- `GET /v1/health`;
- `GET /v1/checkpoints`;
- `POST /v1/games` with `protocolVersion`, black/white checkpoint IDs and `mctsSims`.

The API can be preserved independently of the old training stack. The current backend migration blockers are explicit:

1. `integration/models.py` imports `alphazero.NNetWrapper` and loads `.pkl` checkpoints through `NNetWrapper.from_checkpoint`;
2. it also resolves legacy game/model metadata for old descriptors;
3. `integration/generation.py` creates `GenericPlayers.MCTSPlayer`, which keeps the old MCTS adapter in the product path.

Therefore the next GoCube migration should replace the backend model-loader/player adapter behind Protocol V1, not change the frontend contract.

## 8. Arena boundary

Current Arena remains standalone and separate from training:

```text
tools/arena.py
  -> tools/arena_engine.py
  -> tools/arena_profiles registry
  -> tools/arena_profiles/torus9.py
  -> current Torus9 profile/contract
  -> gocube_golden.torus9 evaluation implementation
```

The current Torus9 Arena contract is 64 minimum games, 64 simulations, cpuct 1.25, FPU 0, root noise OFF, temperature 0, fast search OFF, resign OFF, watchdog 1000, workers 16, batched ON, Arena batch size 8, inference wait 6 ms, paired starts/color swap ON, technical outcomes fail-closed/excluded, gating decoupled, komi 0.5. Old Arena entrypoints are not authorized as current by this boundary.

## 9. Regression/guard coverage added by this boundary

Focused tests lock:

- exactly one Torus9 profile with `status=current`;
- canonical profile path and sanitized current content fingerprint;
- komi 0.5 in rules/self-play/Arena and rejection of a noncanonical explicit value;
- 80×8 network and exact head shapes;
- observation, target and self-play fingerprints;
- self-play/replay/training budgets;
- retired profiles absent from current launcher/runtime selection;
- current launcher/runtime do not import `Coach`, `SelfPlayAgent` or `NNetWrapper`;
- standalone Torus9 Arena adapter resolves the current profile/contract;
- the forbidden historical decimal komi literal is absent from the current production path;
- immutable reference fingerprint remains provenance-only and is not confused with the sanitized current content hash.

CI still runs the repository-wide pytest and existing smoke/differential suites. No long training run is added.

## 10. Remaining legacy dependencies / next removal candidates

Next-stage cleanup can now be concrete:

1. migrate Protocol V1 model loading from `NNetWrapper` to a current Golden checkpoint/model adapter while preserving endpoints and JSON schema;
2. migrate Protocol V1 game generation off `GenericPlayers.MCTSPlayer`/legacy Cython MCTS if the product boundary still requires live generation;
3. retire old Torus9 profiles and experiment launchers once no test/integration import remains;
4. retire old Arena/checkpoint/gating runners now superseded by `tools/arena.py`;
5. remove `Coach`/`SelfPlayAgent` and legacy CI build obligations only after their remaining consumers are proven absent;
6. separately redesign self-play execution/process parallelism to address Legion under-utilization, without altering this scientific contract.

## 11. Boundary conclusion

`M0→M17` is reproducibly identifiable by immutable Git/run provenance and a single authoritative current Torus9 profile. The current training/resume path, standalone Arena path and GoCube product boundary are separated and classified. The only profile-content change in this PR is removal of forbidden historical sentinel metadata; the resulting whole-file fingerprint changes, while every scientific sub-contract remains fixed.

**Scientific semantics changed: NO.**
