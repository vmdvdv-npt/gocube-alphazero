from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gocube_golden.artifact_resolver as artifact_resolver
from gocube_golden.cube_m0_publisher import publish_cube_m0
from gocube_golden.provenance import CodeIdentity
import tools.publish_cube_m0 as publish_tool
from tools.cube4_production_qualification import (
    _qualification_acceptance,
    _require_acceptance,
)


def _valid_selfplay() -> dict[str, object]:
    return {
        "move_limit": 600,
        "games": 64,
        "formal_games": 64,
        "technical_games": 0,
        "technical_reasons": {},
        "invalid_records": 0,
        "worker_errors": [],
        "central_inference_fatal": None,
    }


def _valid_arena() -> dict[str, object]:
    return {
        "games_requested": 64,
        "games_valid": 64,
        "technical_games": 0,
        "invalid_games": 0,
        "paired_colors": {"candidate_black": 32, "candidate_white": 32},
    }


def _assert_qualification_rejected(
    selfplay: dict[str, object], arena: dict[str, object]
) -> None:
    acceptance = _qualification_acceptance(selfplay, arena)
    assert acceptance["passed"] is False
    with pytest.raises(RuntimeError, match="failed acceptance checks"):
        _require_acceptance(acceptance, "synthetic qualification")


def test_qualification_rejects_technical_selfplay() -> None:
    selfplay = _valid_selfplay()
    selfplay.update(
        formal_games=63,
        technical_games=1,
        technical_reasons={"WORKER_ERROR": 1},
    )
    _assert_qualification_rejected(selfplay, _valid_arena())


def test_qualification_rejects_final_move_limit_after_fallbacks() -> None:
    selfplay = _valid_selfplay()
    selfplay.update(
        move_limit=1600,
        formal_games=63,
        technical_games=1,
        technical_reasons={"MOVE_LIMIT": 1},
    )
    _assert_qualification_rejected(selfplay, _valid_arena())


def test_qualification_rejects_technical_arena() -> None:
    arena = _valid_arena()
    arena.update(games_valid=63, technical_games=1)
    _assert_qualification_rejected(_valid_selfplay(), arena)


def test_qualification_rejects_invalid_arena() -> None:
    arena = _valid_arena()
    arena.update(games_valid=63, invalid_games=1)
    _assert_qualification_rejected(_valid_selfplay(), arena)


def test_qualification_accepts_only_fully_valid_result() -> None:
    acceptance = _qualification_acceptance(_valid_selfplay(), _valid_arena())
    assert acceptance["passed"] is True
    assert all(acceptance.values())
    _require_acceptance(acceptance, "synthetic qualification")


def _prepare_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(_repo_root: Path, *args: str) -> str:
        if args and args[0] == "status":
            return ""
        return "a" * 40

    monkeypatch.setattr(publish_tool, "_git", fake_git)
    monkeypatch.setattr(publish_tool.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        publish_tool.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=20 * 1024**3),
    )


@pytest.mark.parametrize("size", (2, 4, 7))
def test_publish_preflight_uses_validated_size_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    _prepare_preflight(monkeypatch)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    lineage = "same-lineage"

    assert publish_tool._preflight(
        tmp_path, runs_root, lineage, size, None
    ) == size

    occupied = runs_root / f"cube{size}" / "active" / lineage
    occupied.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="lineage id is already occupied"):
        publish_tool._preflight(tmp_path, runs_root, lineage, size, None)


def test_publish_preflight_rejects_archived_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_preflight(monkeypatch)
    runs_root = tmp_path / "runs"
    lineage = "archived-lineage"
    (runs_root / "cube7" / "archive" / lineage).mkdir(parents=True)

    with pytest.raises(RuntimeError, match="lineage id is already occupied"):
        publish_tool._preflight(tmp_path, runs_root, lineage, 7, None)


def test_publish_preflight_rejects_unsupported_size_before_other_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        publish_tool,
        "_git",
        lambda *_args: (_ for _ in ()).throw(AssertionError("git must not run")),
    )
    with pytest.raises(ValueError):
        publish_tool._preflight(tmp_path, tmp_path / "runs", "bad-size", 8, None)


def _config(size: int = 2) -> dict[str, object]:
    return {
        "schema": "gocube-effective-config-v2",
        "version": 2,
        "topology": f"cube{size}",
        "compatibility": {
            "family": "cube-v2",
            "size": size,
            "topology": f"cube{size}",
        },
        "self_play": {"master_seed": 2026092301},
        "training": {
            "learning_rate": 0.001,
            "batch_size": 1,
            "optimizer_steps": 1,
            "weight_decay": 0.0,
        },
        "replay": {"generations": 2, "cap": 16},
        "execution": {},
        "arena": {},
        "supervision": {},
        "extensions": {"master_seed": 2026092301},
    }


def _identity() -> CodeIdentity:
    return CodeIdentity("0" * 40, "1" * 40, True)


def test_m0_has_no_fallible_resolver_readback_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    lineage = "cube2-no-postcommit-readback"
    final_root = runs_root / "cube2" / "active" / lineage
    original_checkpoint = artifact_resolver.ArtifactResolver.checkpoint

    def fail_if_already_published(self, ref):
        if final_root.exists():
            raise RuntimeError("synthetic post-publication resolver failure")
        return original_checkpoint(self, ref)

    monkeypatch.setattr(
        artifact_resolver.ArtifactResolver,
        "checkpoint",
        fail_if_already_published,
    )

    publication = publish_cube_m0(
        size=2,
        lineage_id=lineage,
        effective_config=_config(),
        seed=2026092301,
        runs_root=runs_root,
        code_identity=_identity(),
    )

    assert publication.root == final_root
    assert final_root.is_dir()


def test_m0_resolver_failure_happens_before_commit_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    lineage = "cube2-precommit-resolver-failure"
    final_root = runs_root / "cube2" / "active" / lineage

    def fail_before_commit(self, _ref):
        assert not final_root.exists()
        raise RuntimeError("synthetic resolver failure")

    monkeypatch.setattr(
        artifact_resolver.ArtifactResolver,
        "checkpoint",
        fail_before_commit,
    )

    with pytest.raises(RuntimeError, match="synthetic resolver failure"):
        publish_cube_m0(
            size=2,
            lineage_id=lineage,
            effective_config=_config(),
            seed=2026092301,
            runs_root=runs_root,
            code_identity=_identity(),
        )

    assert not final_root.exists()
    active_root = runs_root / "cube2" / "active"
    assert not list(active_root.glob(f".{lineage}.m0-publishing-*"))


def test_m0_commit_creates_one_canonical_lineage_and_is_immutable(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    lineage = "cube2-single-canonical-m0"
    publication = publish_cube_m0(
        size=2,
        lineage_id=lineage,
        effective_config=_config(),
        seed=2026092301,
        runs_root=runs_root,
        code_identity=_identity(),
    )

    assert publication.root == runs_root / "cube2" / "active" / lineage
    assert publication.root.is_dir()
    assert list(runs_root.rglob("M0.pt")) == [publication.root / "checkpoints" / "M0.pt"]
    active_root = runs_root / "cube2" / "active"
    assert not list(active_root.glob(f".{lineage}.m0-publishing-*"))

    with pytest.raises(FileExistsError, match="existing Cube M0 lineage"):
        publish_cube_m0(
            size=2,
            lineage_id=lineage,
            effective_config=_config(),
            seed=2026092301,
            runs_root=runs_root,
            code_identity=_identity(),
        )
