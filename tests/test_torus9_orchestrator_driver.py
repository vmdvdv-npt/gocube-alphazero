from __future__ import annotations

import json
from pathlib import Path

from gocube_golden.orchestrator import OrchestratorSpec
from gocube_golden.execution_reference import LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
from tools.torus9_orchestrator_driver import (
    PERIODIC_ARENA_PRESET_FINGERPRINT,
    PERIODIC_ARENA_STARTSET_FINGERPRINT,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "gocube" / "torus9_training_orchestrator_v1.json"


def test_torus9_orchestrator_spec_is_concrete_and_current():
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rendered = json.dumps(payload, sort_keys=True)
    assert "<" not in rendered and "profile-arena-driver" not in rendered
    assert payload["topology"] == "torus9"
    assert payload["expected_profile_fingerprint"] == (
        "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"
    )
    assert payload["arena"]["every_generations"] == 5
    assert payload["arena"]["preset_fingerprint"] == PERIODIC_ARENA_PRESET_FINGERPRINT
    assert payload["arena"]["startset_fingerprint"] == PERIODIC_ARENA_STARTSET_FINGERPRINT
    spec = OrchestratorSpec.load(SPEC_PATH, repo_root=ROOT)
    assert spec.topology == "torus9"
    assert spec.arena_required is True


def test_torus9_orchestrator_generation_command_pins_validated_legion_selfplay():
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    argv = payload["execution"]["generation_command"]
    values = {argv[index]: argv[index + 1] for index in range(len(argv) - 1) if argv[index].startswith("--")}
    reference = LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
    assert int(values["--workers"]) == reference.recommended_workers
    assert int(values["--active-games-per-worker"]) == reference.recommended_active_games_per_worker
    assert int(values["--total-active-contexts"]) == reference.recommended_total_active_contexts
    assert int(values["--batch-cap"]) == reference.recommended_batch_cap
    assert float(values["--wait-ms"]) == reference.recommended_wait_ms
    assert values["--device"] == "cuda"


def test_cube_is_not_claimed_as_a_current_orchestrator_driver():
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    commands = payload["execution"]["generation_command"] + payload["arena"]["command"]
    assert "tools/torus9_orchestrator_driver.py" in commands
    assert all("cube" not in token.lower() for token in commands)
