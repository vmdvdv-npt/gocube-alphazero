from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.orchestrator_v2 import (
    ArenaRunRequest,
    ArenaRunner,
    ResolvedCheckpointNode,
    torus9_startset_ref,
)
from gocube_golden.orchestrator_v2.contracts import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfigRef,
)
from tools.arena import _canonical_evaluation_output
from tools.arena_engine import ArenaExecutionConfig


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _resolved(tmp_path: Path, lineage: str, generation: int, sha: str) -> ResolvedCheckpointNode:
    path = tmp_path / lineage / "checkpoints" / f"M{generation}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint-{lineage}-{generation}".encode())
    checkpoint = CheckpointRef(
        "torus9",
        lineage,
        f"M{generation}",
        generation,
        f"checkpoints/M{generation}.pt",
        sha,
    )
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=EffectiveConfigRef(
            ArtifactRef("metadata/effective.json", SHA_A),
            SHA_B,
        ),
        provenance=ArtifactRef("metadata/provenance.json", SHA_A),
    )
    return ResolvedCheckpointNode(
        node=node,
        checkpoint=SimpleNamespace(path=path),  # type: ignore[arg-type]
        effective_config=SimpleNamespace(),  # type: ignore[arg-type]
        provenance=SimpleNamespace(),  # type: ignore[arg-type]
        owner_root=path.parents[1],
        owner_status="ACTIVE",
    )


def _config() -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
        games=16,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cpu",
        strict_production=False,
        min_mean_inference_batch_rows=0.0,
        min_effective_cpu_cores=0.0,
        early_gate_enabled=False,
    )


def _request(tmp_path: Path) -> ArenaRunRequest:
    return ArenaRunRequest(
        candidate=_resolved(tmp_path, "candidate", 95, SHA_A),
        reference=_resolved(tmp_path, "reference", 90, SHA_B),
        master_seed=202609131004,
        startset=torus9_startset_ref(master_seed=202609131004, games=16),
        config=_config(),
    )


def _fake_engine(**kwargs: object) -> dict[str, object]:
    output = kwargs["output_dir"]
    assert isinstance(output, Path)
    run_id = str(kwargs["run_id"])
    summary = {
        "games": 16,
        "W/L/D": [8, 8, 0],
        "telemetry": {
            "technical_games": 0,
            "performance_status": "HEALTHY",
            "performance_failures": [],
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return summary


def test_runner_forwards_resolved_paths_and_persists_full_identity(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def engine(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _fake_engine(**kwargs)

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, run_id: tmp_path / "evaluations" / run_id,
    )
    request = _request(tmp_path)
    result = ArenaRunner(engine=engine).run(request)

    assert captured["candidate_path"] == request.candidate.path
    assert captured["reference_path"] == request.reference.path
    assert result.wld == (8, 8, 0)
    assert result.validity == "VALID"
    assert result.identity.candidate.to_dict() == request.candidate.ref.to_dict()
    assert result.identity.reference.to_dict() == request.reference.ref.to_dict()
    assert result.identity.startset.to_dict() == request.startset.to_dict()
    assert result.identity.games == 16
    assert (result.output_dir / "evaluation-identity.json").is_file()
    assert not (result.output_dir / "checkpoints").exists()


def test_runner_reuses_only_matching_complete_result(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def engine(**kwargs: object) -> dict[str, object]:
        calls.append(1)
        return _fake_engine(**kwargs)

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, run_id: tmp_path / "evaluations" / run_id,
    )
    runner = ArenaRunner(engine=engine)
    request = _request(tmp_path)
    first = runner.run(request)
    second = runner.run(request)

    assert calls == [1]
    assert first.evaluation_id == second.evaluation_id
    assert second.validity == "VALID"


def test_runner_reclaims_markerless_stale_directory(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def engine(**kwargs: object) -> dict[str, object]:
        calls.append(1)
        return _fake_engine(**kwargs)

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, run_id: tmp_path / "evaluations" / run_id,
    )
    runner = ArenaRunner(engine=engine)
    request = _request(tmp_path)
    stale = tmp_path / "evaluations" / runner._evaluation_id(request)
    stale.mkdir(parents=True)
    (stale / "runtime").mkdir()
    (stale / "runtime" / "telegram-notifications.jsonl").write_text(
        "stale\n", encoding="utf-8"
    )

    result = runner.run(request)

    assert calls == [1]
    assert result.validity == "VALID"
    assert (result.output_dir / "evaluation-identity.json").is_file()


def test_runner_can_store_same_lineage_result_under_generation_directory(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []
    captured: dict[str, object] = {}

    def engine(**kwargs: object) -> dict[str, object]:
        calls.append(1)
        captured.update(kwargs)
        return _fake_engine(**kwargs)

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda *_args: (_ for _ in ()).throw(AssertionError("same-lineage Arena must not use evaluations/")),
    )
    request = replace(
        _request(tmp_path),
        reference=_resolved(tmp_path, "candidate", 94, SHA_B),
        output_dir=tmp_path / "candidate" / "arena" / "generation-0095",
    )
    runner = ArenaRunner(engine=engine)

    first = runner.run(request)
    second = runner.run(request)

    assert calls == [1]
    assert first.output_dir.parent == request.output_dir
    assert first.output_dir.name == first.evaluation_id
    assert second.output_dir == first.output_dir
    assert captured["allowed_lineage_arena_root"] == (
        request.candidate.owner_root / "arena"
    ).resolve()


def test_runner_rejects_custom_output_for_cross_lineage_evaluation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same-lineage"):
        ArenaRunner(engine=_fake_engine).run(
            replace(
                _request(tmp_path),
                output_dir=tmp_path / "candidate" / "arena" / "generation-0095",
            )
        )


def test_runner_rejects_same_lineage_output_outside_lineage_arena(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the lineage arena directory"):
        ArenaRunner(engine=_fake_engine).run(
            replace(
                _request(tmp_path),
                reference=_resolved(tmp_path, "candidate", 94, SHA_B),
                output_dir=tmp_path / "outside-arena",
            )
        )


def test_production_arena_boundary_allows_only_explicit_lineage_arena_root(
    tmp_path: Path,
) -> None:
    lineage_arena = tmp_path / "active" / "lineage" / "arena"
    same_lineage_output = lineage_arena / "generation-0095" / "evaluation-id"

    assert _canonical_evaluation_output(
        "torus9",
        same_lineage_output,
        allowed_lineage_arena_root=lineage_arena,
    ) == same_lineage_output.resolve()

    with pytest.raises(ValueError, match="canonical runs"):
        _canonical_evaluation_output(
            "torus9",
            tmp_path / "active" / "other-lineage" / "arena" / "evaluation-id",
            allowed_lineage_arena_root=lineage_arena,
        )


def test_runner_reuses_completed_non_valid_arena_and_preserves_boundary_status(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def engine(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["run_id"]))
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        identity = kwargs["evaluation_identity"]
        assert isinstance(identity, dict)
        status = str(kwargs["comparison"])
        summary = {
            "games": 16,
            "W/L/D": [7, 8, 1],
            "validity": status,
            "telemetry": {"technical_games": 1 if status == "TECHNICAL" else 0},
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output / "manifest.json").write_text(
            json.dumps({"run_id": str(kwargs["run_id"])}), encoding="utf-8"
        )
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "candidate": identity["candidate"],
                    "reference": identity["reference"],
                    "profile": "torus9",
                    "master_seed": identity["master_seed"],
                }
            ),
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(
        "gocube_golden.orchestrator_v2.arena_runner.evaluation_dir",
        lambda _topology, run_id: tmp_path / "evaluations" / run_id,
    )
    request = _request(tmp_path)
    runner = ArenaRunner(engine=engine)

    for offset, status in enumerate(("TECHNICAL", "CRITICAL", "INVALID"), start=1):
        variant = replace(
            request,
            master_seed=request.master_seed + offset,
            startset=torus9_startset_ref(
                master_seed=request.master_seed + offset,
                games=request.config.games,
            ),
            comparison=status,
        )
        first = runner.run(variant)
        second = runner.run(variant)
        assert first.validity == second.validity == status

    assert len(calls) == 3


def test_identity_changes_for_seed_startset_and_full_arena_contract(tmp_path: Path) -> None:
    request = _request(tmp_path)
    base = ArenaRunner._identity(request).fingerprint
    assert ArenaRunner._identity(replace(request, master_seed=request.master_seed + 1)).fingerprint != base
    assert ArenaRunner._identity(
        replace(request, startset=torus9_startset_ref(master_seed=7, games=16))
    ).fingerprint != base
    changed = dict(request.execution_contract or {})
    changed["inference_batch_wait_ms"] = 2.0
    assert ArenaRunner._identity(replace(request, execution_contract=changed)).fingerprint != base
