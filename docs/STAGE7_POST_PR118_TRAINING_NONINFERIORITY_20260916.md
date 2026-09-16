# Stage 7 — post-PR118 Torus9 training non-inferiority

- Run: `torus9-stage7-post-pr118-20260916-run04`
- Implementation SHA: `c99fd38013861febca7e15ded0412a6b4bac9f89`
- Profile: `gocube-torus9-golden-v3` / `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`
- Final status: **PASS**

## Historical reference

OLD lineage: `torus9-golden-v3-20260914-run03` (resolved through CheckpointCatalog; read-only).

| Stage | Model SHA | Artifact SHA-256 |
|---:|---|---|
| M0 | `sha256:e11d31217e8cecf89294405203c6ef62372f3478c68b9cc7a2f1666e10c2c285` | `sha256:2da9b40576f67eba2fbf4afa1fa7287a71151f1f379723d358cfa2a8bb91c800` |
| M5 | `sha256:9d0c591464c729eb8fff880bd1b719aa4ed78d76b6bb85d62f7adf8106e04bd1` | `sha256:4b0bde834de658d6e5bebf3c584d7951843b58df209f95df52a8824a3bf8e6f2` |
| M8 | `sha256:1638519dfd639934f9b61cde36df4265545821d9fc00c35c2858c5d3f001305f` | `sha256:3e1fd5000e17c888541833eac781ec24e426cc1dab725e142a92f2e5b5470e86` |
| M10 | `sha256:f71ee1d742c8c8eeb719276c703c690f73eb103941291214943e359d5ef4e8d8` | `sha256:a811ebac6cf21b369528379ca1e63293a3d536d522fc4fc8017bce649b631ef0` |
| M14 | `sha256:15fcb2e70ebbf1ea250d91d578f18f8799cf529ad399d5d341f248c44e60540f` | `sha256:05cc0182670e2bda3f599c6138c7dc14507bdb3d149699d85501106be739bf93` |
| M17 | `sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5` | `sha256:86722afe70fefd1d4a2a408e47c3492c7b6da43e86f283888d8815da5b037e53` |

## New lineage and iteration accounting

NEW starts from the OLD M0 reference; the checkpoint was not copied. The M3 process was stopped and the same run resumed for M4.

| Stage | Games | Technical | Positions | Replay | Adam step | Model SHA | Wall sec | Moves/s |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| M1 | 64 | 0 | 4711 | 4711 | 80 | `sha256:95f08d20f7d0ff6f4e436632eb051d8dbc586f546c0b2eda487a59a18f906e42` | 273.7747040009999 | 25.133695028685754 |
| M2 | 64 | 0 | 6826 | 11537 | 160 | `sha256:dea179c1f2c5019d8ee9f08e565513b0cb232e656794385113b261d19ccfcd98` | 423.4244201709953 | 27.542301785542577 |
| M3 | 64 | 0 | 7022 | 18559 | 240 | `sha256:806f424489f6843c416b6e180fdf7cee47ddc2049852188c40a000a77f7ed527` | 501.21130443199945 | 26.398978094996465 |
| M4 | 64 | 0 | 7072 | 20000 | 320 | `sha256:a79da657b551449c13e699b1ca8a90ea434e6a57ab32c0131b8bfa526116303f` | 526.0245858039998 | 26.286884935942254 |
| M5 | 64 | 0 | 6385 | 20000 | 400 | `sha256:cdadead183cd9fc5afcdbd9dba80311eb7b878eb4304ef032da42a109054c758` | 508.02569257900177 | 25.29665454894345 |

## Resume test

`{"before": {"adam_step": 240, "boundary": "M3", "checkpoint_sha256": "sha256:6c83ccbe322dcdcd782d260f22cd85062e991479b7a82ddddb9ad9188e0bebcd", "metadata_sha256": "sha256:af9b121eadc7990984daa2430a757ab6ec9494140f0e7ca8218e5a13003e4d87", "optimizer_updates": 240, "published": true, "replay_fingerprint": "sha256:ab21510b12c3680802954f5f52640cbcaedb46dab5d94893fc33a09f8a47b36a", "replay_rows": 18559, "replay_sha256": "sha256:05d2014804acddc04e1fdf145393dbdba82ca474bd88b0e3f6ed422a8b5be4a7", "run_id": "torus9-stage7-post-pr118-20260916-run04"}, "boundary": "M3", "checkpoint_parent": {"artifact_sha256": "sha256:6c83ccbe322dcdcd782d260f22cd85062e991479b7a82ddddb9ad9188e0bebcd", "label": "M3", "metadata_path": "/home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-stage7-post-pr118-20260916-run04/checkpoints/M3.metadata.json", "model_hash": "sha256:806f424489f6843c416b6e180fdf7cee47ddc2049852188c40a000a77f7ed527", "path": "/home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-stage7-post-pr118-20260916-run04/checkpoints/M3.pt"}, "generation_not_repeated_or_skipped": true, "optimizer_state_continuous": true, "replay_recovered": true, "same_run_id": true, "status": "PASS"}`

## Performance

PR110 → PR112 → post-PR118: PR110 validated standard-64 ≈20.931 moves/s; PR112 exact reproduction 21.223284 moves/s; post-PR118 measured below

Post-PR118 median moves/s: **26.286885**; verdict: **PASS**.

## Same-stage Arena ladder

| Stage | Comparison | Games | NEW W/L/D | NEW score | 95% cluster CI | Verdict |
|---:|---|---:|---|---:|---|---|
| M5 | NEW M5 vs OLD M5 | 192 | 113/79/0 | 0.588542 | [0.531250, 0.645833] | **PASS** |

## Final verdicts

`GOLDEN SCIENTIFIC CONTRACT: PASS`
`FRESH POST-PR118 TRAINING: PASS`
`SELF-PLAY TECHNICAL GAMES: 0`
`TRAINING ACCOUNTING: PASS`
`REPLAY CONTINUITY: PASS`
`STOP/RESUME: PASS`
`CHECKPOINT INTEGRITY: PASS`
`POST-PR118 PERFORMANCE: PASS`
`LAST SAME-STAGE COMPARISON: NEW M5 vs OLD M5`
`LAST ARENA GAMES: 192`
`STRENGTH NON-INFERIORITY: PASS`
`HISTORICAL LINEAGE MUTATED: NO`
`KOMI: 0.5`
`RUN STORAGE POLICY: PASS`
`GOLDEN PARAMETERS CHANGED: NO`

`STAGE 7: PASS`
