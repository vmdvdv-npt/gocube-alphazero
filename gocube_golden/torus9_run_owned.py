"""Run-owned Torus9 tuning boundary.

The Golden profile remains an immutable reference for scientific invariants,
but four operator-controlled knobs are deliberately outside Golden equality
checks: learning rate, replay window/cap, self-play MCTS simulations, and
periodic Arena cadence. Their effective values remain part of run provenance.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from . import torus9_contract as _contract
from . import torus9_monolith as _core

RUN_OWNED_TUNABLES = (
    "training.learning_rate",
    "replay.generations",
    "replay.cap",
    "self_play.mcts_simulations",
    "arena.every_generations",
)

_ORIGINAL_PROFILE_LOADER = _contract.load_torus9_current_profile
_ORIGINAL_PROFILE_VALIDATOR = _contract.validate_torus9_current_profile
_PROFILE_LOADER_INSTALLED = False
_RUN_SPEC_INSTALLED = False
_TELEGRAM_INSTALLED = False
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _merge(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return result


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def validate_run_owned_profile(
    profile: Mapping[str, Any],
    *,
    base_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate Golden invariants without comparing the run-owned knobs.

    The four knobs are checked only for basic executable shape/range. They are
    never compared with Golden values and therefore cannot cause a Golden-drift
    error merely because the operator selected another valid value.
    """
    if profile.get("profile_id") != _contract.TORUS9_CURRENT_PROFILE_ID:
        raise ValueError("Run-owned Torus9 profile requires the current profile id")
    marker = profile.get("experiment")
    if marker is not None and not isinstance(marker, Mapping):
        raise ValueError("Run-owned Torus9 experiment marker must be an object")

    self_play = profile.get("self_play")
    training = profile.get("training")
    replay = profile.get("replay")
    if not isinstance(self_play, Mapping) or not isinstance(training, Mapping) or not isinstance(replay, Mapping):
        raise ValueError("Run-owned Torus9 profile is missing self_play/training/replay")

    _positive_int(self_play.get("mcts_simulations"), "self_play.mcts_simulations")
    _positive_float(training.get("learning_rate"), "training.learning_rate")
    _positive_int(replay.get("generations"), "replay.generations")
    _optional_positive_int(replay.get("cap"), "replay.cap")

    base = dict(base_profile) if base_profile is not None else _ORIGINAL_PROFILE_LOADER()
    normalized = deepcopy(dict(profile))
    normalized.pop("experiment", None)
    normalized.pop("base_profile_path", None)

    # Replace only operator-owned tunables with the immutable reference values
    # while validating every other Golden invariant through the original
    # machine-checked validator.
    normalized["self_play"]["mcts_simulations"] = base["self_play"]["mcts_simulations"]
    normalized["self_play"]["fingerprint"] = base["self_play"]["fingerprint"]
    normalized["training"]["learning_rate"] = base["training"]["learning_rate"]
    normalized["replay"]["window"] = base["replay"]["window"]
    normalized["replay"]["generations"] = base["replay"]["generations"]
    normalized["replay"]["cap"] = base["replay"]["cap"]
    normalized["content_fingerprint"] = base["content_fingerprint"]
    normalized["profile_fingerprint"] = base["profile_fingerprint"]
    _ORIGINAL_PROFILE_VALIDATOR(normalized, verify_fingerprint=True)
    return dict(profile)


def validate_torus9_run_profile(
    profile: Mapping[str, Any],
    *,
    verify_fingerprint: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Public validator that treats the four tuning knobs as run-owned."""
    if not isinstance(profile, Mapping):
        raise ValueError("Current Torus 9×9 profile must be a JSON object")
    if isinstance(profile.get("experiment"), Mapping):
        validated = validate_run_owned_profile(profile)
        if verify_fingerprint:
            expected_content = _contract.current_torus9_content_fingerprint(profile)
            if profile.get("content_fingerprint") != expected_content:
                raise ValueError(f"Current Torus 9×9 content fingerprint mismatch: {expected_content}")
            expected_profile = _contract.profile_fingerprint(profile)
            if profile.get("profile_fingerprint") != expected_profile:
                raise ValueError(f"Experimental Torus 9×9 profile fingerprint drift: {expected_profile}")
        return validated
    return _ORIGINAL_PROFILE_VALIDATOR(
        profile,
        verify_fingerprint=verify_fingerprint,
        repo_root=repo_root,
    )


def load_torus9_run_profile(
    path: str | Path | None = None,
    *,
    verify_fingerprint: bool = True,
) -> dict[str, Any]:
    """Resolve a Torus9 profile with run-owned tuning overlays.

    Canonical profiles still use the original strict loader. An overlay may
    change only fields that survive ``validate_run_owned_profile``; changing LR,
    replay, or self-play simulations never requires a Golden-value exception.
    """
    if path is None:
        return _ORIGINAL_PROFILE_LOADER(None, verify_fingerprint=verify_fingerprint)
    profile_path = Path(path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Torus9 profile must be a JSON object")
    base_ref = raw.get("base_profile_path")
    if not base_ref:
        return _ORIGINAL_PROFILE_LOADER(profile_path, verify_fingerprint=verify_fingerprint)

    base_path = (profile_path.parent / str(base_ref)).resolve()
    base = _ORIGINAL_PROFILE_LOADER(base_path, verify_fingerprint=True)
    overlay = dict(raw)
    overlay.pop("base_profile_path", None)
    profile = _merge(dict(base), overlay)
    validate_run_owned_profile(profile, base_profile=base)

    # The complete run profile is still fingerprinted for provenance. The
    # fingerprint is descriptive, not a Golden allow/deny gate.
    profile["content_fingerprint"] = _contract.current_torus9_content_fingerprint(profile)
    profile["profile_fingerprint"] = _contract.profile_fingerprint(profile)
    return profile


def install_profile_loader() -> None:
    global _PROFILE_LOADER_INSTALLED
    if _PROFILE_LOADER_INSTALLED:
        return
    _contract.load_torus9_current_profile = load_torus9_run_profile
    _contract.validate_torus9_current_profile = validate_torus9_run_profile
    _PROFILE_LOADER_INSTALLED = True


class RunOwnedTorus9SelfPlaySearchContract(_core.Torus9SelfPlaySearchContract):
    """Canonical Torus9 search semantics with operator-owned simulation count."""

    def validate(self) -> None:
        actual = asdict(self)
        expected = asdict(_core.Torus9SelfPlaySearchContract())
        simulations = actual.pop("simulations")
        expected.pop("simulations")
        _positive_int(simulations, "self_play.mcts_simulations")
        if actual != expected:
            raise ValueError("Torus 9×9 self-play search contract drift outside run-owned MCTS simulations")


def install_selfplay_boundary(module: object) -> None:
    """Remove only Golden-value equality for run-owned self-play knobs."""
    def validate_boundary(
        model: object,
        *,
        profile_id: str,
        profile_fp: str,
        contract: object,
    ) -> None:
        if float(_contract.TORUS9_KOMI) != 0.5:
            raise RuntimeError("Current Torus9 self-play requires komi 0.5")
        if profile_id != _contract.TORUS9_CURRENT_PROFILE_ID:
            raise ValueError("Torus9 self-play supports only the current profile family")
        if not isinstance(model, _core.Torus9CurrentGraphNet):
            raise ValueError("Current Torus9 self-play requires GoldenGraphNetV2-Torus9")
        if int(model.hidden) != _contract.TORUS9_CURRENT_HIDDEN or int(model.blocks_count) != _contract.TORUS9_CURRENT_BLOCKS:
            raise ValueError("Current Torus9 network must remain 80x8")
        if getattr(contract, "contract_id", None) != _contract.TORUS9_CURRENT_SELFPLAY_CONTRACT_ID:
            raise ValueError("Current Torus9 self-play contract id drift")
        if not math.isclose(
            float(getattr(contract, "dirichlet_alpha")),
            float(_contract.TORUS9_CURRENT_DIRICHLET_ALPHA),
            abs_tol=0.0,
        ):
            raise ValueError("Current Torus9 Dirichlet alpha drift")
        # Production resolves and binds the exact full run-profile fingerprint
        # before entering self-play.  This lower boundary therefore checks its
        # integrity/shape, not equality to the Golden reference fingerprint.
        if not _SHA256_RE.fullmatch(str(profile_fp)):
            raise ValueError("Current Torus9 profile fingerprint drift")
        getattr(contract, "validate")()

    setattr(module, "_validate_current_scientific_boundary", validate_boundary)


def _run_owned_tunables_from_spec(spec: object) -> dict[str, object]:
    profile = getattr(getattr(spec, "orchestrator_spec"), "profile_payload")
    payload = getattr(spec, "payload")
    self_play = profile["self_play"]
    training = profile["training"]
    replay = profile["replay"]
    arena = payload["arena"]
    return {
        "learning_rate": float(training["learning_rate"]),
        "replay_generations": int(replay["generations"]),
        "replay_cap": replay["cap"],
        "replay_window": str(replay.get("window", "")),
        "self_play_mcts_simulations": int(self_play["mcts_simulations"]),
        "arena_every_generations": int(arena["every_generations"]),
    }


def install_run_spec_policy() -> None:
    """Make Torus9 profile fingerprint descriptive for run-owned knobs.

    The resolved full profile fingerprint is still persisted in the lineage.
    A stale static ``expected_profile_fingerprint`` cannot reject a Torus9 run
    merely because an operator changed one of the run-owned tuning values.
    """
    global _RUN_SPEC_INSTALLED
    if _RUN_SPEC_INSTALLED:
        return
    from . import run_spec as module

    original_resolve = module._resolve_profile
    original_create = module.StrictProductionTrainingOrchestrator.create

    def resolve_profile(repo_root: Path, payload: Mapping[str, object]):
        profile_ref = str(payload.get("profile_path", "")).strip()
        if profile_ref:
            relative = Path(profile_ref)
            candidate = (repo_root / relative).resolve() if not relative.is_absolute() else relative
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, Mapping) and raw.get("profile_id") == _contract.TORUS9_CURRENT_PROFILE_ID:
                if relative.is_absolute() or ".." in relative.parts or repo_root.resolve() not in candidate.parents:
                    raise ValueError("profile_path must be repo-relative and remain inside repository root")
                profile = load_torus9_run_profile(candidate)
                return candidate, profile, str(profile["profile_fingerprint"])
        return original_resolve(repo_root, payload)

    def create(self: object, *, parent_checkpoint: Mapping[str, object] | None = None) -> None:
        original_create(self, parent_checkpoint=parent_checkpoint)
        strict_spec = getattr(self, "strict_run_spec")
        orchestrator_spec = getattr(strict_spec, "orchestrator_spec")
        # This policy is Torus9-specific.  The universal orchestrator must keep
        # working unchanged for Cube and future adapters.
        if getattr(orchestrator_spec, "topology", None) != "torus9":
            return
        manifest_path = getattr(self, "paths").manifest
        manifest = module.read_json(manifest_path)
        manifest["operator_tunables"] = _run_owned_tunables_from_spec(strict_spec)
        module.atomic_write_json(manifest_path, manifest)

    module._resolve_profile = resolve_profile
    module.StrictProductionTrainingOrchestrator.create = create
    _RUN_SPEC_INSTALLED = True


def install_telegram_start_notification() -> None:
    global _TELEGRAM_INSTALLED
    if _TELEGRAM_INSTALLED:
        return
    from . import telegram_notifier as tg

    original = tg.build_notification

    def build_notification(
        paths: object,
        level: str,
        message: str,
        details: Mapping[str, object],
    ):
        if message == "Training orchestrator started":
            manifest = tg._json(Path(getattr(paths, "manifest")))
            tunables = manifest.get("operator_tunables")
            if isinstance(tunables, Mapping):
                lineage = str(manifest.get("lineage_id") or Path(getattr(paths, "root")).name)
                lr = float(tunables["learning_rate"])
                replay_generations = int(tunables["replay_generations"])
                replay_cap = tunables["replay_cap"]
                mcts = int(tunables["self_play_mcts_simulations"])
                cadence = int(tunables["arena_every_generations"])
                key = f"training-start:{lineage}:{manifest.get('config_fingerprint', '')}"
                text = "\n".join(
                    [
                        "Training started — GoCube AlphaZero",
                        f"Lineage: {lineage}",
                        f"LR: {lr:g}",
                        f"Replay: {replay_generations} generations / {replay_cap} positions",
                        f"Self-play MCTS: {mcts} sims",
                        f"Arena cadence: every {cadence} generations",
                    ]
                )
                return key, text
        return original(paths, level, message, details)

    tg.build_notification = build_notification
    _TELEGRAM_INSTALLED = True


__all__ = [
    "RUN_OWNED_TUNABLES",
    "RunOwnedTorus9SelfPlaySearchContract",
    "install_profile_loader",
    "install_run_spec_policy",
    "install_selfplay_boundary",
    "install_telegram_start_notification",
    "load_torus9_run_profile",
    "validate_run_owned_profile",
    "validate_torus9_run_profile",
]
