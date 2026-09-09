# GoCube model integration contract

GoCube model compatibility is resolved once at the training boundary.  The
concrete training game class and its effective network arguments produce the
`gocube-model-contract-v2` object.  Contract version 2 adds the explicit
`semanticGameVariant` identity; version 1 checkpoints are accepted only when
their persisted exact `gameClassId` proves that identity.  The same object is recorded in
`gocube-run.json` as `modelContract`, in the rich `run-manifest.json` and
`effective-config.json` artifacts as `model_contract`, and as flat
`gocube_*` fields plus `gocube_model_contract` in checkpoint args.

The contract includes:

- rules implementation, terminal adjudicator, rules fingerprint, and komi;
- semantic game variant (`plain`, `pinned`, or `diversified_pinned`);
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

During V1-to-V2 migration, the loader also recomputes the exact V1 network
architecture fingerprint. Plain `train.py` V1 checkpoints are validated
against the pre-save hash (before the derived `gocube_network_architecture`
field was copied into checkpoint args); production profile classes retain
their explicit class-level architecture ID in that legacy computation. A V1
fingerprint is accepted only when this legacy hash matches exactly.

## Production model profiles

The production KataGo training path has one explicit architecture selector:
`--model-profile baseline` or `--model-profile g1`.  The default is the
reproducible `baseline` profile, so G1 is not an implicit production default.
Both profiles use the same training, self-play, replay, and evaluation path;
the selector records the architecture and structural-channel contract in the
effective configuration and resume manifest.

The baseline profile keeps the historical graph architecture, observation
shape `(18, 96, 1)` for Cube 4, architecture ID `gocube-graph-v1`, and zero
structural channels.  The explicit G1 profile uses shape `(20, 96, 1)`,
architecture ID `gocube-graph-structural-v1`, and two structural channels.
Future B0/B1 profiles should be added to this same selector rather than
creating a second production training path.

`tests/fixtures/gocube_production_profile_baseline.json` is an architecture,
search-budget, optimizer, and training-hyperparameter reference for the
baseline profile.  Its `architecture_reference_commit` is not a complete
rules snapshot, and its `reference_scope` makes that boundary explicit.  The
fixture intentionally does not freeze the local rules implementation version;
the final B0 freeze happens only after S1, S2, S3, H1, and M1 are integrated.

The compact manifest is authoritative only when it contains the complete
contract.  The loader reads checkpoint metadata before constructing a network,
resolves the exact compatible game class, compares it with the catalog
descriptor and all available rich artifacts, and only then loads weights.
Conflicts fail closed with the field, saved value, and expected value.  A
topology/size-only resolver is reserved for explicit historical V1/V2/V3
manifests, where the terminal-adjudicator version is part of the legacy
semantics.

## Semantic-game restoration

The supported semantic identities are:

- `plain`: the direct Japanese V3 training class;
- `pinned`: the pinned pass/endgame observation and episode semantics;
- `diversified_pinned`: pinned semantics plus the production synthetic-start
  diversification curriculum.

Baseline and G1 are model/profile identities over the same
`diversified_pinned` semantic game.  Arena first reads the checkpoint
contract, resolves topology and size, selects the class factory named by
`semanticGameVariant`, and recomputes the semantic contract.  It then compares
rules implementation and fingerprint, action schema/size, topology and all
point/adjacency fingerprints, terminal adjudicator, semantic variant, and
komi before starting a game.  Observation/model profile compatibility is
checked independently for each loaded network.  Thus a plain checkpoint is
never opened as diversified merely because it is Cube 3 or Cube 4, and an
ambiguous legacy record is rejected instead of guessed.

To add a semantic variant, add a stable variant token to the contract schema,
register its concrete class factory in the contract resolver, set the token on
the class, and add matrix plus negative tests.  Do not infer it from topology,
observation-channel count, run name, or filename.

The explicit G1 profile records the structural observation contract with shape
`(20, 96, 1)` for Cube 4 and architecture ID
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
