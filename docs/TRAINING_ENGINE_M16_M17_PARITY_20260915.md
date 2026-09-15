# Training Engine M16→M17 parity

Date: 2026-09-15

The Stage-3 training boundary was run against the immutable historical
Torus9 Golden artifacts. The reproduction used M16, persisted replay, and the
persisted generation-17 fresh rows only. No self-play was run and the
canonical run namespace was not written.

| Field | Evidence |
|---|---|
| source checkpoint | M16 |
| source model hash | `sha256:4d4df8572d04b0f383261d69e7aeac3825fbffc294a2310d00e20e35801ade26` |
| source optimizer step | 1280 |
| canonical Golden base commit | `53946d0c84fca5a6f81a387bfd399ea62e34b088` |
| reproduced M17 base commit | `53946d0c84fca5a6f81a387bfd399ea62e34b088` |
| fresh generation | 17 |
| fresh positions | 6424 |
| rolling replay positions | 20000 |
| training seed | `1208412472300244515` |
| optimizer steps | 80 |
| samples consumed | 5120 |

## Results

- Model hash: **PASS** — reproduced and canonical M17 both have `sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5`.
- Adam state: **PASS** — 1360 steps, exact normalized state parity, maximum absolute tensor delta `0.0`.
- Fresh replay: **PASS** — 6424 rows, exact ordering and file SHA `sha256:fe75ea13055e6765f78970d7c950ae930d37920735883c2c3281e7fcc90b4377`.
- Rolling replay: **PASS** — 20000 rows, exact ordering and replay fingerprint `sha256:aac53717c0459973f6e707e76e6a73664df3ae049ae9e8a16d677f25e805f40f`.
- Sampling: **PASS** — source counts `M15=1574`, `M16=1877`, `M17=1669`; all 80 batches are size 64.
- Metadata lineage: **PASS** — exact equality for `base_commit`, parent identity, self-play contract, profile, target, checkpoint label, and run identity.
- Canonical M17 mutation: **NO**.
- M18 created: **NO**.

Semantic validation covered the training-visible rolling generations 14–16
and generation 17. Older fresh artifacts were parsed post-by-post and used in
the exact generation reconstruction/eviction evidence; they cannot enter the
M17 training window.

Scientific semantics changed: **NO**
Training semantics changed: **NO**
Replay semantics changed: **NO**
Checkpoint schema changed: **NO**
Komi: **0.5**
Profile fingerprint: `sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775`
Target fingerprint: `sha256:02ab244688534b271473302ab4edf00516b91b8c9e592d2a8de63dfb5`
