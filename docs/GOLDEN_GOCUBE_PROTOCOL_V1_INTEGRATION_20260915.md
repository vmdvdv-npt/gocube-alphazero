# Golden → GoCube Protocol V1 integration

Status: backend implementation on `codex/golden-gocube-protocol-v1-stage4`.

## Before

```text
iteration-XXXX.pkl
→ legacy NNetWrapper
→ legacy GoCube game
→ GenericPlayers.MCTSPlayer
→ Protocol V1
```

## After

```text
MXX.pt + MXX.metadata.json
→ GoldenCheckpointLoader
→ Golden evaluator
→ canonical SequentialPUCT
→ GoldenState/rules/apply_action/scoring
→ Protocol V1 adapter
```

The legacy catalog, loader, and generator remain available as a compatibility
path. Golden dispatch is selected from the validated descriptor backend kind;
there is no fallback from a failed Golden load to the legacy wrapper.

## Supported Golden profiles

Only these current, metadata-identified profiles are playable:

| Profile | Topology | Points/actions | Network |
| --- | --- | ---: | --- |
| `gocube-torus9-golden-v3` | `torus-9x9-row-major-v1` | 81 / 82 | `GoldenGraphNetV2-Torus9`, 80×8, WDL + auxiliary heads |
| `gocube-cube4-golden-training-v1` | `cube4x4x6-golden-topology-v1` | 96 / 97 | `GoldenCubeGraphNetV1` |

The loader requires the current profile/schema/fingerprint, topology and
geometry identity, observation and target fingerprints, action/head shapes,
search provenance, model hash, and `komi = 0.5`. Unknown, retired, malformed,
or unpaired `.pt` files are not returned by the public catalog.

## Point/action mapping proof

Mappings are constructed by PointId identity, not by assuming integer order.
The bridge then checks the reverse bijection and, for every point, compares the
set of Golden neighbours mapped to Protocol PointIds with the product topology
neighbour set.

| Topology | Golden points | Protocol PASS | Golden topology fingerprint | Adjacency proof |
| --- | ---: | ---: | --- | --- |
| Cube 4×4×6 | 96 | 96 | `sha256:b98a44c8172b94f522968c9fdf972b88755f6fd601d46dd3902ee473b170a591` | PASS, all 96 points |
| Torus 9×9 | 81 | 81 | `sha256:a417b6d4e3da67ead03240361976a6f412c89403077bec0ac0c0a68ca9665ed1` | PASS, all 81 points |

The Golden topology remains the scientific authority. Product topology is
used only as a read-only boundary audit. Captures come from the Golden
`Transition.captured` tuple and are converted back to Protocol PointIds.
The resulting Cube and Torus mappings also match the existing GoCube product
boundary fixture point-for-point and adjacency-for-adjacency.

## Rules and Protocol V1

Protocol V1 exposes `ruleSet: "chinese"` for area/komi/double-pass UI
projection, while the internal descriptor retains the Golden
`rules_fingerprint` and `terminalAdjudicator: "golden-graph-area-v1"`.
Golden scoring is exact graph-area: black/white area, White komi, no cleanup,
and no invented prisoners or dead-stone diagnostics.

Important compatibility boundary: Golden rules use positional superko, while
the historical GoCube V1 engine specifies simple-ko. Therefore the `chinese`
value is not treated as proof that the two rule engines are interchangeable;
Golden replay and terminal scoring remain authoritative for Golden games. A
future frontend rule label/extension is only needed if the browser must itself
replay or adjudicate arbitrary Golden superko histories.

## Real M17 smoke

Artifact: `torus9-golden-v3-20260914-run03@17` from the immutable local run.

```text
model hash: sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5
profile fingerprint: sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775
rules fingerprint: sha256:e0fd15c82d42a63ca05c3b6fb3ae02deb938543e1e06483ecfeab275dc98a39e
komi: 0.5
requested sims: 1
moves: 74
passes: 2
captures: 24
winner: white
margin: 0.5
all moves legal: PASS
protocol serialization: PASS
Golden round-trip replay: PASS
legacy imports used: NO
canonical checkpoint mutated: NO
M18 created: NO
```

The smoke used `M17` only for inference and did not write into the source run.

## Legacy compatibility

Legacy `.pkl` discovery, metadata validation, `NNetWrapper` loading, and the
old game generator remain unchanged in their compatibility path. Cache keys
include checkpoint ID, device, and backend family, so a textual ID cannot
alias a Golden and legacy model.
