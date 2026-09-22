from __future__ import annotations

from types import SimpleNamespace

from tools.torus9_run_driver import _v2_expected_replay_identity, _v2_replay_scope


def test_v2_replay_scope_accepts_two_generations_without_position_cap() -> None:
    value = SimpleNamespace(config={"replay": {"generations": 2, "cap": None}})

    assert _v2_replay_scope(value) == (2, None)


def test_v2_replay_identity_retains_all_rows_without_position_cap() -> None:
    artifacts = [
        SimpleNamespace(
            sha256=f"sha256:{'1' * 64}",
            identity={
                "generation_identity": {
                    "schema": "torus9-replay-generation-artifact-v1",
                    "generation": 91,
                    "sha256": f"sha256:{'1' * 64}",
                    "row_count": 3,
                }
            },
        ),
        SimpleNamespace(
            sha256=f"sha256:{'2' * 64}",
            identity={
                "generation_identity": {
                    "schema": "torus9-replay-generation-artifact-v1",
                    "generation": 92,
                    "sha256": f"sha256:{'2' * 64}",
                    "row_count": 5,
                }
            },
        ),
    ]

    identity = _v2_expected_replay_identity(artifacts, generations=2, cap=None)

    assert identity is not None
    assert identity["replay_identity_contract"]["maximum_positions"] is None
    assert [
        item["retained_row_count"] for item in identity["generation_identities"]
    ] == [3, 5]
