from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from gocube_golden.artifact_catalog import (
    ARTIFACT_VALIDATION_SCHEMA,
    ArtifactCatalog,
    sha256_file,
)
from gocube_golden.orchestrator_v2.artifact_resolver import (
    ArtifactIntegrityError,
    ArtifactResolutionError,
    ArtifactResolver,
    GraphIntegrityError,
    checkpoint_node_path,
)
from gocube_golden.orchestrator_v2.contracts import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.provenance import canonical_json, sha256_fingerprint
from gocube_golden.torus9_training import TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA
from tools.torus9_run_driver import _resolve_v2_replay_sources


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def _artifact(root: Path, relative: str, content: bytes) -> ArtifactRef:
    path = root / relative
    return ArtifactRef(relative, _write(path, content))


def _config() -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "synthetic", "board": "tiny"},
        self_play={"mcts_simulations": 1},
        training={"learning_rate": 0.1},
        replay={"window": 3, "cap": 8},
        execution={"workers": 1},
        arena={"games": 1},
        supervision={"retries": 0},
    )


@dataclass(frozen=True)
class SyntheticGraph:
    runs_root: Path
    refs: dict[int, CheckpointRef]
    roots: dict[str, Path]


def _make_graph(tmp_path: Path, *, archive_lineage_b: bool = False) -> SyntheticGraph:
    runs_root = tmp_path / "runs"
    states = {"lineage-a": "active", "lineage-b": "archive" if archive_lineage_b else "active"}
    roots = {
        lineage: runs_root / "torus9" / state / lineage
        for lineage, state in states.items()
    }
    for root in roots.values():
        root.mkdir(parents=True)

    config = _config()
    config_bytes = canonical_json(config.to_dict()).encode("utf-8")
    refs: dict[int, CheckpointRef] = {}
    parent_generation = {1: 0, 2: 1, 3: 2, 4: 3}
    owners = {0: "lineage-a", 1: "lineage-a", 2: "lineage-a", 3: "lineage-b", 4: "lineage-b"}
    config_refs: dict[str, EffectiveConfigRef] = {}
    for lineage, root in roots.items():
        config_ref = _artifact(root, "metadata/config/effective.json", config_bytes)
        config_refs[lineage] = EffectiveConfigRef(config_ref, config.fingerprint)

    checkpoint_hashes: dict[str, dict[str, str]] = {lineage: {} for lineage in roots}
    nodes: list[tuple[Path, CheckpointNode]] = []
    for generation in range(5):
        lineage = owners[generation]
        root = roots[lineage]
        checkpoint_path = f"checkpoints/M{generation}.pt"
        checkpoint_sha = _write(root / checkpoint_path, f"checkpoint-M{generation}".encode())
        checkpoint_hashes[lineage][checkpoint_path] = checkpoint_sha
        refs[generation] = CheckpointRef(
            "torus9", lineage, f"M{generation}", generation, checkpoint_path, checkpoint_sha
        )

    for generation in range(5):
        lineage = owners[generation]
        root = roots[lineage]
        checkpoint = refs[generation]
        if generation == 0:
            parent = None
            fresh_replay = None
        else:
            parent = refs[parent_generation[generation]]
            replay_path = f"replay/fresh-M{generation}.jsonl"
            fresh_replay = _artifact(root, replay_path, f"replay-M{generation}".encode())
        provenance = _artifact(root, f"metadata/provenance/M{generation}.json", b"synthetic provenance")
        node = CheckpointNode(
            checkpoint=checkpoint,
            genesis=parent is None,
            parent=parent,
            fresh_replay=fresh_replay,
            effective_config=config_refs[lineage],
            provenance=provenance,
        )
        nodes.append((checkpoint_node_path(root, checkpoint), node))

    for lineage, root in roots.items():
        manifest = {
            "lineage_id": lineage,
            "topology": "torus9",
            "status": "ARCHIVED" if states[lineage] == "archive" else "ACTIVE",
            "parent_checkpoint": None,
            "git_commit": "synthetic",
            "config_fingerprint": config.fingerprint,
            "created_at": "2026-09-18T00:00:00+00:00",
            "checkpoint_hashes": checkpoint_hashes[lineage],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for path, node in nodes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(node.to_dict()), encoding="utf-8")
    return SyntheticGraph(runs_root, refs, roots)


def _install_committed_rolling(
    graph: SyntheticGraph,
    *,
    generation: int = 4,
    generations: int = 3,
    cap: int = 8,
    wrong_rolling_sha: bool = False,
    wrong_composition: bool = False,
) -> tuple[ArtifactResolver, object]:
    """Install V1-compatible catalog evidence for a synthetic parent rolling file."""
    resolver = ArtifactResolver(graph.runs_root)
    parent = resolver.checkpoint(graph.refs[generation])
    fresh = resolver.replay_window(parent, generations)
    fresh_generations = list(range(generation - generations + 1, generation + 1))
    components = [
        {
            "schema": TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
            "generation": fresh_generation,
            "sha256": artifact.sha256,
            "row_count": 1,
            "retained_row_count": 1,
        }
        for fresh_generation, artifact in zip(fresh_generations, fresh)
    ]
    contract = {
        "selection": "rolling-recent-generations-then-last-cap-v1",
        "generations": generations,
        "maximum_positions": cap,
    }
    composition_payload = {
        "schema": "torus9-replay-composition-v1",
        "contract": contract,
        "components": components,
    }
    composition_fingerprint = sha256_fingerprint(composition_payload)
    rolling_path = graph.roots[parent.lineage_id] / "replay" / f"rolling-after-{generation:02d}.jsonl"
    rolling_sha = _write(rolling_path, b'{"source_generation":4}\n')
    rolling_components = [dict(component) for component in components]
    if wrong_composition:
        rolling_components = [
            dict(rolling_components[0], sha256="sha256:" + "f" * 64),
            *rolling_components[1:],
        ]
        composition_payload["components"] = rolling_components
        composition_fingerprint = sha256_fingerprint(composition_payload)
    rolling_entry = {
        "path": rolling_path.relative_to(graph.roots[parent.lineage_id]).as_posix(),
        "sha256": ("sha256:" + "e" * 64) if wrong_rolling_sha else rolling_sha,
        "size_bytes": rolling_path.stat().st_size,
        "row_count": generations,
        "source_generations": list(range(generation - generations + 1, generation + 1)),
        "canonical_replay_fingerprint": composition_fingerprint,
        "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
        "replay_identity_schema": "torus9-replay-composition-v1",
        "generation_identities": rolling_components,
        "replay_identity_contract": contract,
    }
    for lineage, root in graph.roots.items():
        catalog = ArtifactCatalog.initialize(
            root / "runtime/artifact-catalog.json",
            lineage_id=lineage,
            root=root,
        )
        catalog_artifacts: dict[int, list[dict[str, object]]] = {}
        for fresh_generation, artifact in zip(fresh_generations, fresh):
            if artifact.owner_lineage_id != lineage:
                continue
            component = next(
                item for item in components if int(item["generation"]) == fresh_generation
            )
            catalog_artifacts.setdefault(int(component["generation"]), []).append(
                {
                    "path": artifact.ref.path,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.path.stat().st_size,
                    "generation_identity": component,
                }
            )
        if lineage == parent.lineage_id:
            catalog_artifacts.setdefault(generation, []).append(rolling_entry)
        for catalog_generation, artifacts in sorted(catalog_artifacts.items()):
            catalog.register_generation(catalog_generation, artifacts)
    config = EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "synthetic", "board": "tiny"},
        replay={"generations": generations, "cap": cap},
    )
    return resolver, (parent, config, rolling_path, rolling_entry)


def _install_marker_backed_rolling(
    graph: SyntheticGraph,
    *,
    marker_row_count: int | None = None,
) -> tuple[object, object]:
    resolver, (parent, config, rolling_path, rolling_entry) = _install_committed_rolling(graph)
    root = graph.roots[parent.lineage_id]
    marker = {
        "generation": parent.generation,
        "checkpoint_sha256": parent.ref.sha256,
        "rolling_replay_sha256": rolling_entry["sha256"],
        "replay_row_count": (
            rolling_entry["row_count"] if marker_row_count is None else marker_row_count
        ),
        "replay_generations": rolling_entry["source_generations"],
        "replay_fingerprint": rolling_entry["canonical_replay_fingerprint"],
        "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
        "replay_identity_schema": rolling_entry["replay_identity_schema"],
        "replay_identity_components": rolling_entry["generation_identities"],
        "replay_identity_contract": rolling_entry["replay_identity_contract"],
    }
    marker_path = root / f"generation-{parent.generation:02d}.complete.json"
    marker_sha = _write(marker_path, canonical_json(marker).encode("utf-8"))
    provenance_path = parent.provenance.path
    provenance = {
        "generation_commit": {
            "path": marker_path.relative_to(root).as_posix(),
            "sha256": marker_sha,
        }
    }
    provenance_sha = _write(provenance_path, canonical_json(provenance).encode("utf-8"))
    original = CheckpointNode.from_dict(
        json.loads(checkpoint_node_path(root, parent.ref).read_text())
    )
    _rewrite_node(
        graph,
        parent.generation,
        CheckpointNode(
            checkpoint=original.checkpoint,
            genesis=original.genesis,
            parent=original.parent,
            fresh_replay=original.fresh_replay,
            effective_config=original.effective_config,
            provenance=ArtifactRef(
                provenance_path.relative_to(root).as_posix(), provenance_sha
            ),
        ),
    )
    refreshed = ArtifactResolver(graph.runs_root).checkpoint(parent.ref)
    return refreshed, config


def _rewrite_node(graph: SyntheticGraph, generation: int, node: CheckpointNode) -> None:
    target = graph.refs[generation]
    path = checkpoint_node_path(graph.roots[target.lineage_id], target)
    path.write_text(json.dumps(node.to_dict()), encoding="utf-8")


def test_checkpoint_opens_same_lineage_and_archived_owner(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path, archive_lineage_b=True)
    resolver = ArtifactResolver(graph.runs_root)

    same = resolver.checkpoint(graph.refs[2])
    archived = resolver.checkpoint(graph.refs[4])

    assert same.checkpoint.path == graph.roots["lineage-a"] / "checkpoints/M2.pt"
    assert archived.owner_status == "ARCHIVED"
    assert archived.effective_config.fingerprint == _config().fingerprint
    assert archived.provenance.path.read_bytes() == b"synthetic provenance"


def test_checkpoint_requires_canonical_node_and_matches_identity(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver = ArtifactResolver(graph.runs_root)
    node_path = checkpoint_node_path(graph.roots["lineage-a"], graph.refs[2])
    node_path.unlink()
    with pytest.raises(ArtifactResolutionError, match="canonical CheckpointNode"):
        resolver.checkpoint(graph.refs[2])

    graph = _make_graph(tmp_path / "identity")
    wrong = CheckpointNode(
        checkpoint=graph.refs[1],
        genesis=False,
        parent=graph.refs[0],
        fresh_replay=ArtifactRef("replay/fresh-M1.jsonl", graph.refs[1].sha256),
        effective_config=EffectiveConfigRef(
            ArtifactRef("metadata/config/effective.json", sha256_file(graph.roots["lineage-a"] / "metadata/config/effective.json")),
            _config().fingerprint,
        ),
        provenance=ArtifactRef(
            "metadata/provenance/M1.json",
            sha256_file(graph.roots["lineage-a"] / "metadata/provenance/M1.json"),
        ),
    )
    _rewrite_node(graph, 2, wrong)
    with pytest.raises(GraphIntegrityError, match="identity mismatch"):
        resolver = ArtifactResolver(graph.runs_root)
        resolver.checkpoint(graph.refs[2])


def test_checkpoint_wrong_physical_sha_fails_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    checkpoint_path = graph.roots["lineage-a"] / "checkpoints/M2.pt"
    checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="actual sha256"):
        ArtifactResolver(graph.runs_root).checkpoint(graph.refs[2])


def test_parent_genesis_same_lineage_and_cross_lineage(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver = ArtifactResolver(graph.runs_root)

    genesis = resolver.checkpoint(graph.refs[0])
    assert resolver.parent(genesis) is None
    child = resolver.checkpoint(graph.refs[4])
    parent = resolver.parent(child)
    assert parent is not None
    assert parent.ref == graph.refs[3]
    assert resolver.parent(parent).ref == graph.refs[2]  # type: ignore[union-attr]


def test_ancestor_and_ancestors_order_and_bounds(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver = ArtifactResolver(graph.runs_root)
    child = resolver.checkpoint(graph.refs[4])

    assert resolver.ancestor(child, 0).ref == graph.refs[4]
    assert resolver.ancestor(child, 1).ref == graph.refs[3]
    assert resolver.ancestor(child, 3).ref == graph.refs[1]
    assert [item.ref for item in resolver.ancestors(child, 4)] == [
        graph.refs[3], graph.refs[2], graph.refs[1], graph.refs[0]
    ]
    assert resolver.ancestors(child, 0) == ()
    with pytest.raises(GraphIntegrityError, match="ancestry ended"):
        resolver.ancestor(child, 5)
    with pytest.raises(GraphIntegrityError, match="ancestry ended"):
        resolver.ancestors(child, 5)


def test_parent_generation_gap_is_rejected(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    original = CheckpointNode.from_dict(
        json.loads(
            checkpoint_node_path(graph.roots["lineage-b"], graph.refs[4]).read_text()
        )
    )
    broken = CheckpointNode(
        checkpoint=original.checkpoint,
        genesis=False,
        parent=graph.refs[2],
        fresh_replay=original.fresh_replay,
        effective_config=original.effective_config,
        provenance=original.provenance,
    )
    _rewrite_node(graph, 4, broken)
    with pytest.raises(GraphIntegrityError, match="generation mismatch"):
        ArtifactResolver(graph.runs_root).parent(ArtifactResolver(graph.runs_root).checkpoint(graph.refs[4]))


def test_cycle_is_controlled_and_not_an_infinite_loop(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    nodes: dict[int, CheckpointNode] = {}
    for generation in range(5):
        root = graph.roots["lineage-a" if generation < 3 else "lineage-b"]
        nodes[generation] = CheckpointNode.from_dict(
            json.loads(checkpoint_node_path(root, graph.refs[generation]).read_text())
        )
    for child, parent_generation in ((1, 2), (2, 1), (3, 2)):
        node = nodes[child]
        nodes[child] = CheckpointNode(
            checkpoint=node.checkpoint,
            genesis=False,
            parent=graph.refs[parent_generation],
            fresh_replay=node.fresh_replay,
            effective_config=node.effective_config,
            provenance=node.provenance,
        )
    for generation in (1, 2, 3):
        _rewrite_node(graph, generation, nodes[generation])

    resolver = ArtifactResolver(graph.runs_root)
    with pytest.raises(GraphIntegrityError, match="cycle"):
        resolver.ancestor(resolver.checkpoint(graph.refs[1]), 3)


def test_replay_window_is_graph_only_and_oldest_to_newest(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver = ArtifactResolver(graph.runs_root)
    parent = resolver.checkpoint(graph.refs[4])
    window = resolver.replay_window(parent, 4)
    assert [artifact.path.read_bytes() for artifact in window] == [
        b"replay-M1", b"replay-M2", b"replay-M3", b"replay-M4"
    ]
    assert [artifact.owner_lineage_id for artifact in window] == [
        "lineage-a", "lineage-a", "lineage-b", "lineage-b"
    ]
    assert resolver.replay_window(parent, 0) == ()


def test_replay_window_attaches_catalog_and_checkpoint_evidence(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    root = graph.roots["lineage-b"]
    fresh_path = root / "replay/iter-04-fresh.jsonl"
    fresh_path.write_text('{"source_generation": 4}\n', encoding="utf-8")
    fresh_sha = sha256_file(fresh_path)
    original = CheckpointNode.from_dict(
        json.loads(checkpoint_node_path(root, graph.refs[4]).read_text())
    )
    _rewrite_node(
        graph,
        4,
        CheckpointNode(
            checkpoint=original.checkpoint,
            genesis=original.genesis,
            parent=original.parent,
            fresh_replay=ArtifactRef("replay/iter-04-fresh.jsonl", fresh_sha),
            effective_config=original.effective_config,
            provenance=original.provenance,
        ),
    )
    component = {
        "schema": TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
        "generation": 4,
        "sha256": fresh_sha,
        "row_count": 1,
    }
    (root / "checkpoints/M4.metadata.json").write_text(
        json.dumps({
            "replay_generation_identities": [component],
            "replay_identity_schema": "torus9-replay-composition-v1",
            "replay_identity_contract": {"generations": 1, "maximum_positions": 1},
            "replay_fingerprint": "sha256:" + "4" * 64,
        }),
        encoding="utf-8",
    )
    catalog = ArtifactCatalog.initialize(
        root / "runtime/artifact-catalog.json",
        lineage_id="lineage-b",
        root=root,
    )
    catalog.register_generation(
        4,
        [{"path": "replay/iter-04-fresh.jsonl", "sha256": fresh_sha, "size_bytes": fresh_path.stat().st_size}],
    )

    artifact = ArtifactResolver(graph.runs_root).replay_window(
        ArtifactResolver(graph.runs_root).checkpoint(graph.refs[4]),
        1,
    )[0]

    assert artifact.identity is not None
    assert artifact.identity["immutable_verified"] is True
    assert artifact.identity["validation_schema"] == ARTIFACT_VALIDATION_SCHEMA
    # The generic resolver authenticates the referenced file and catalog
    # evidence, but does not interpret Torus9 replay composition policy.
    assert "generation_identity" not in artifact.identity


def test_replay_window_missing_and_corrupt_artifacts_fail_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    replay = graph.roots["lineage-b"] / "replay/fresh-M4.jsonl"
    replay.unlink()
    resolver = ArtifactResolver(graph.runs_root)
    with pytest.raises(ArtifactIntegrityError, match="expected SHA"):
        resolver.replay_window(resolver.checkpoint(graph.refs[4]), 1)

    graph = _make_graph(tmp_path / "corrupt")
    replay = graph.roots["lineage-a"] / "replay/fresh-M2.jsonl"
    replay.write_bytes(b"corrupt")
    with pytest.raises(ArtifactIntegrityError, match="expected SHA"):
        ArtifactResolver(graph.runs_root).replay_window(
            ArtifactResolver(graph.runs_root).checkpoint(graph.refs[4]), 3
        )


def test_compatible_parent_rolling_replay_is_selected_as_one_restore_artifact(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, config, rolling_path, _entry) = _install_committed_rolling(graph)

    paths, shas, identities, replay_identity = _resolve_v2_replay_sources(parent, config)

    assert paths == (rolling_path.resolve(),)
    assert len(shas) == len(identities) == 1
    assert replay_identity["replay_identity_contract"] == {
        "selection": "rolling-recent-generations-then-last-cap-v1",
        "generations": 3,
        "maximum_positions": 8,
    }


def test_marker_backed_parent_rolling_replay_skips_fresh_window_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _make_graph(tmp_path)
    parent, config = _install_marker_backed_rolling(graph)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("trusted V2 rolling replay must not resolve fresh sources")

    monkeypatch.setattr(ArtifactResolver, "replay_window", forbidden)
    paths, _shas, _identities, replay_identity = _resolve_v2_replay_sources(parent, config)

    assert paths == (
        (graph.roots[parent.lineage_id] / "replay/rolling-after-04.jsonl").resolve(),
    )
    assert replay_identity["row_count"] == 3
    assert replay_identity["commit_artifact"]["path"].endswith("generation-04.complete.json")


def test_marker_row_count_corruption_fails_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    parent, config = _install_marker_backed_rolling(graph, marker_row_count=2)

    with pytest.raises(ArtifactIntegrityError, match="row count"):
        _resolve_v2_replay_sources(parent, config)


def test_changed_replay_scope_falls_back_to_graph_fresh_window(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, _config, _rolling_path, _entry) = _install_committed_rolling(graph)
    changed = EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "synthetic", "board": "tiny"},
        replay={"generations": 2, "cap": 8},
    )

    paths, _shas, _identities, _replay_identity = _resolve_v2_replay_sources(parent, changed)

    assert len(paths) == 2
    assert all("fresh-M" in path.name for path in paths)


def test_changed_replay_cap_falls_back_to_graph_fresh_window(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, _config, _rolling_path, _entry) = _install_committed_rolling(graph)
    changed = EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "synthetic", "board": "tiny"},
        replay={"generations": 3, "cap": 7},
    )

    paths, _shas, _identities, _replay_identity = _resolve_v2_replay_sources(parent, changed)

    assert len(paths) == 3
    assert all("fresh-M" in path.name for path in paths)


def test_rolling_replay_wrong_sha_fails_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, config, _rolling_path, _entry) = _install_committed_rolling(
        graph, wrong_rolling_sha=True
    )

    with pytest.raises(ArtifactIntegrityError, match="expected SHA"):
        _resolve_v2_replay_sources(parent, config)


def test_rolling_replay_wrong_composition_fails_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, config, _rolling_path, _entry) = _install_committed_rolling(
        graph, wrong_composition=True
    )

    with pytest.raises(ArtifactIntegrityError, match="composition"):
        _resolve_v2_replay_sources(parent, config)


def test_cross_lineage_parent_rolling_replay_is_referenced_without_copy(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver, (parent, config, rolling_path, _entry) = _install_committed_rolling(graph)
    child_root = graph.roots["lineage-a"] / "child-output"

    paths, _shas, _identities, _replay_identity = _resolve_v2_replay_sources(parent, config)

    assert parent.lineage_id == "lineage-b"
    assert paths[0] == rolling_path.resolve()
    assert not (child_root / rolling_path.name).exists()


def test_effective_config_sha_fingerprint_and_topology_are_verified(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    config_path = graph.roots["lineage-a"] / "metadata/config/effective.json"
    config_path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="expected SHA"):
        ArtifactResolver(graph.runs_root).checkpoint(graph.refs[2])

    graph = _make_graph(tmp_path / "fingerprint")
    config_path = graph.roots["lineage-a"] / "metadata/config/effective.json"
    changed = _config().to_dict()
    changed["training"] = {"learning_rate": 0.2}
    config_path.write_text(canonical_json(changed), encoding="utf-8")
    original = CheckpointNode.from_dict(
        json.loads(checkpoint_node_path(graph.roots["lineage-a"], graph.refs[2]).read_text())
    )
    _rewrite_node(
        graph,
        2,
        CheckpointNode(
            checkpoint=original.checkpoint,
            genesis=original.genesis,
            parent=original.parent,
            fresh_replay=original.fresh_replay,
            effective_config=EffectiveConfigRef(
                ArtifactRef(
                    original.effective_config.artifact.path,
                    sha256_file(config_path),
                ),
                original.effective_config.fingerprint,
            ),
            provenance=original.provenance,
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="fingerprint mismatch"):
        ArtifactResolver(graph.runs_root).checkpoint(graph.refs[2])

    graph = _make_graph(tmp_path / "topology")
    config_path = graph.roots["lineage-a"] / "metadata/config/effective.json"
    wrong_topology = dict(_config().to_dict())
    wrong_topology["topology"] = "cube4"
    wrong_config = EffectiveConfig.from_dict(wrong_topology)
    config_path.write_text(canonical_json(wrong_config.to_dict()), encoding="utf-8")
    original = CheckpointNode.from_dict(
        json.loads(checkpoint_node_path(graph.roots["lineage-a"], graph.refs[2]).read_text())
    )
    _rewrite_node(
        graph,
        2,
        CheckpointNode(
            checkpoint=original.checkpoint,
            genesis=original.genesis,
            parent=original.parent,
            fresh_replay=original.fresh_replay,
            effective_config=EffectiveConfigRef(
                ArtifactRef(original.effective_config.artifact.path, sha256_file(config_path)),
                wrong_config.fingerprint,
            ),
            provenance=original.provenance,
        ),
    )
    with pytest.raises(GraphIntegrityError, match="topology mismatch"):
        ArtifactResolver(graph.runs_root).checkpoint(graph.refs[2])


def test_artifact_owner_confinement_is_fail_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    resolver = ArtifactResolver(graph.runs_root)
    owner = resolver.checkpoint(graph.refs[2])
    outside = tmp_path / "outside.bin"
    outside_sha = _write(outside, b"outside")
    link = graph.roots["lineage-a"] / "escape"
    link.symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError, match="escapes owner lineage"):
        resolver.open_artifact(ArtifactRef("escape", outside_sha), owner=owner)
    with pytest.raises(ValueError, match="safe relative path"):
        ArtifactRef("../lineage-b/replay.bin", outside_sha)


def test_missing_parent_checkpoint_fails_closed(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    (graph.roots["lineage-a"] / "checkpoints/M2.pt").unlink()
    resolver = ArtifactResolver(graph.runs_root)
    with pytest.raises(ArtifactIntegrityError, match="exists: false"):
        resolver.ancestor(resolver.checkpoint(graph.refs[4]), 2)
