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

The current pinned training builder records the V4 observation contract with
shape `(18, 96, 1)`.  This shape is metadata-derived and is not a universal
assumption for historical models.  Historical checkpoints remain loadable only
under their actual saved semantics; no metadata is renamed in place.

Inference and training curriculum are separate.  Service and evaluation use
the resolved observation/rules/action/topology/network/search contract, but do
not enable training-only synthetic starts or diversification behavior.
