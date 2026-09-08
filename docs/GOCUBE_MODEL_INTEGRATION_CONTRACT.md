# GoCube model integration contract

GoCube model compatibility is resolved once at the training boundary.  The
concrete training game class and its effective network arguments produce the
`gocube-model-contract-v1` object.  The same object is recorded in
`gocube-run.json` as `modelContract`, in the rich `run-manifest.json` and
`effective-config.json` artifacts as `model_contract`, and as flat
`gocube_*` fields plus `gocube_model_contract` in checkpoint args.

The contract includes:

- rules implementation, terminal adjudicator, rules fingerprint, and komi;
- observation schema and shape;
- action schema and action count;
- topology kind, size, point count, canonical PointId-order fingerprint, and
  adjacency fingerprint;
- network architecture ID and architecture fingerprint;
- search contract ID, target schemas, and output-head count.

`point_order_fingerprint` is the SHA-256 digest of the canonical logical
PointId sequence.  `adjacency_fingerprint` is the digest of the neighbor table
in that sequence.  They are deterministic and do not use Python's process-
dependent `hash()`.

The compact manifest is authoritative only when it contains the complete
contract.  The loader reads checkpoint metadata before constructing a network,
resolves the exact compatible game class, compares it with the catalog
descriptor and all available rich artifacts, and only then loads weights.
Conflicts fail closed with the field, saved value, and expected value.  A
topology/size-only resolver is reserved for explicit historical V1/V2/V3
manifests, where the terminal-adjudicator version is part of the legacy
semantics.

The current pinned training builder records the G1 structural observation
contract with shape `(20, 96, 1)` for Cube 4 and architecture ID
`gocube-graph-structural-v1`.  The first 18 channels retain the pinned V3
observation; channels 18 and 19 are, respectively, binary graph-triangle
membership and normalized shortest-path distance to the triangle point set.
The feature matrix is computed once per topology from canonical adjacency and
is cached; it is independent of stones and contains no PointId embedding.

For a topology with triangles, normalization is
`distance / max(1, max_distance)`, where `max_distance` is the largest BFS
distance in that topology.  A topology without triangles gets zeros in both
channels.  Cube 4 has 8 triangles and 24 participating points; Cube 2--7 and
the supported Torus topologies use the same graph algorithm without a
Cube-4-specific special case.

G1 is rotation/permutation equivariant: all point-indexed inputs and outputs
use the same canonical permutation, while value, score, and PASS remain
invariant.  `gocube-observation-v5-structural-features` and
`gocube-graph-structural-v1` are new identities.  Historical V1/V2/V3/V4
models keep their original identities and shapes.  Old checkpoints and replay
observations are not exact-resume compatible with G1; no input-weight padding
or silent replay migration is performed.

Inference and training curriculum are separate.  Service and evaluation use
the resolved observation/rules/action/topology/network/search contract, but do
not enable training-only synthetic starts or diversification behavior.
