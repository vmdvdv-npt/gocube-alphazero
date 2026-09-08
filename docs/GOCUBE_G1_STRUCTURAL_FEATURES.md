# G1 structural Cube features

G1 removes the empty-Cube expressivity limitation of the shared graph message
network.  The explicit `--model-profile g1` production profile contains the
existing dynamic game-state channels plus two deterministic point channels:

1. `triangle_membership`: `1` when the point belongs to at least one graph
   3-cycle, otherwise `0`;
2. `normalized_distance_to_triangle`: multi-source BFS distance to the set of
   triangle points, divided by `max(1, max_distance)` for that topology.

Triangles are canonicalized sorted triples and are derived only from
`Topology.neighbors_by_index`.  The result is cached by the complete canonical
point order and adjacency table.  A graph without triangles receives zeros in
both channels.  There is no renderer input, face label, coordinate, learned
PointId identity, or strategic corner bonus.

The explicit G1 production profile identities are:

- observation schema: `gocube-observation-v5-structural-features`;
- architecture: `gocube-graph-structural-v1`;
- structural feature schema: `gocube-structural-features-v1`.

Cube 4 has 8 vertex triangles and 24 participating logical points.  Cube 2--7
and Torus 9 use the same adjacency-only algorithm.  The two channels are
point-indexed in the same canonical order as actions, ownership, and the S2
point-order fingerprint.

The common production training path defaults to the historical `baseline`
profile (`--model-profile baseline`), which retains the `(18, 96, 1)`
observation, `gocube-graph-v1`, and zero structural channels.  G1 is selected
explicitly and is recorded in the effective configuration and resume manifest.

For Cube 4 the maximum BFS distance is 2, so its non-zero distance values are
`0.5` and `1.0` after normalization.  Cube 2 has every point in a vertex
triangle and therefore has maximum distance 0.

Graph message passing remains shared and its depth, width, pooling, and heads
are unchanged.  Because structural channels and the message graph are
permuted together, point policy and ownership are equivariant under the 24
canonical Cube rotations; value, score, and PASS are invariant.  G1 proves
structural expressivity only.  It makes no claim about playing strength.

G1 checkpoints and replay tensors are incompatible with historical
architectures because the observation shape and architecture identity change.
Training starts from scratch; no automatic checkpoint weight transfer or
mass replay migration is provided.
