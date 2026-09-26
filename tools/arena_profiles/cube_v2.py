"""Cube V2 scientific adapter for the common board-agnostic Arena engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.cube_arena_contract_v2 import CubeArenaSearchConfig
from gocube_golden.cube_arena_startset_v1 import (
    GENERATOR_VERSION,
    STARTSET_SCHEMA,
    build_cube_arena_startset,
)
from gocube_golden.cube_checkpoint_v2 import CHECKPOINT_SCHEMA, file_sha256
from gocube_golden.cube_family import cube_family_topology, initial_cube_state
from gocube_golden.cube_game_contract_v2 import (
    concrete_game_fingerprint,
    concrete_game_identity,
    load_contract,
)
from gocube_golden.cube_network_v2 import (
    ARCHITECTURE_FINGERPRINT,
    ARCHITECTURE_ID,
    build_cube_model_from_metadata,
    cube_graphnet_v2_model_hash,
    validate_cube_model_metadata,
)
from gocube_golden.cube_observation_v2 import CHANNEL_COUNT, concrete_observation_identity
from gocube_golden.provenance import derive_seed
from tools.arena_engine import ArenaExecutionConfig, CheckpointIdentity
from tools.arena_inference import infer_policy_wdl_batch


# This is the execution preset qualified for the first Cube4 production run.
# It is intentionally kept in the shared Cube family adapter so Cube2..Cube7
# receive the same operational admission rules without importing Torus code.
PRODUCTION_WORKERS = 16
PRODUCTION_GAMES = 192
PRODUCTION_GAMES_PER_WORKER = 12
PRODUCTION_INFERENCE_BATCH_ROWS = 64
PRODUCTION_INFERENCE_BATCH_WAIT_MS = 4.0


def _same_sha(actual: str, expected: str) -> bool:
    return str(actual).removeprefix("sha256:") == str(expected).removeprefix("sha256:")


def _profile_id(size: int, search: CubeArenaSearchConfig) -> str:
    search.validate()
    return (
        f"cube-v2|size={int(size)}|simulations={int(search.simulations)}|"
        f"cpuct={float(search.cpuct):.17g}|fpu={float(search.fpu):.17g}|"
        f"watchdog={int(search.watchdog)}"
    )


def _parse_profile_id(value: str) -> tuple[int, CubeArenaSearchConfig]:
    parts = str(value).split("|")
    if not parts or parts[0] != "cube-v2":
        raise ValueError(f"Invalid Cube V2 Arena profile id: {value!r}")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Malformed Cube V2 Arena profile field: {part!r}")
        key, raw = part.split("=", 1)
        if key in fields:
            raise ValueError(f"Duplicate Cube V2 Arena profile field: {key}")
        fields[key] = raw
    if set(fields) != {"size", "simulations", "cpuct", "fpu", "watchdog"}:
        raise ValueError("Cube V2 Arena profile id is incomplete")
    size = int(fields["size"])
    search = CubeArenaSearchConfig(
        simulations=int(fields["simulations"]),
        cpuct=float(fields["cpuct"]),
        fpu=float(fields["fpu"]),
        watchdog=int(fields["watchdog"]),
    )
    search.validate()
    return size, search


class CubeV2ArenaProfile:
    """Cube-only scientific semantics plugged into the existing Arena runtime."""

    wdl_size = 3
    last_infer_timing: Mapping[str, float] = {}

    def __init__(self, *, size: int, search_config: CubeArenaSearchConfig) -> None:
        topology = cube_family_topology(size)
        search_config.validate()
        self.size = topology.size
        self.topology = topology
        self.search_config = search_config
        state = initial_cube_state(size=self.size)
        self.expected_game_fingerprint = concrete_game_fingerprint(
            concrete_game_identity(
                load_contract(),
                self.size,
                topology_id=topology.topology_id,
                topology_fingerprint=topology.fingerprint,
                rules_fingerprint=state.rules_fingerprint,
                komi=state.komi,
            )
        )
        self.expected_observation_fingerprint = str(
            concrete_observation_identity(topology)["concrete_observation_fingerprint"]
        )
        self.profile_id = _profile_id(self.size, search_config)
        self.run_id_prefix = f"cube{self.size}-arena"
        self.worker_process_prefix = f"arena-cube{self.size}-worker"
        self.observation_shape = (CHANNEL_COUNT, topology.point_count)
        self.policy_size = topology.action_count
        self._last_startset = None

    @classmethod
    def from_profile_id(cls, value: str) -> "CubeV2ArenaProfile":
        size, search = _parse_profile_id(value)
        return cls(size=size, search_config=search)

    def validate_execution_config(self, config: ArenaExecutionConfig) -> None:
        config.validate_base()
        if not config.strict_production:
            return
        if type(config.monitoring_acceptance) is not bool:
            raise ValueError("Cube Arena monitoring_acceptance must be boolean")
        if torch.device(config.device).type != "cuda":
            raise ValueError("Production Cube Arena requires CUDA central inference")
        if config.workers != PRODUCTION_WORKERS:
            raise ValueError(
                "Production Cube Arena requires the qualified 16-worker preset"
            )
        if not config.monitoring_acceptance and config.games < PRODUCTION_GAMES:
            raise ValueError(
                f"Production Cube Arena requires at least {PRODUCTION_GAMES} games"
            )
        if config.games_per_worker != PRODUCTION_GAMES_PER_WORKER:
            raise ValueError(
                "Production Cube Arena requires the qualified 12-games-per-worker preset"
            )
        if config.inference_batch_rows != PRODUCTION_INFERENCE_BATCH_ROWS:
            raise ValueError(
                "Production Cube Arena requires inference_batch_rows=64"
            )
        if float(config.inference_batch_wait_ms) != PRODUCTION_INFERENCE_BATCH_WAIT_MS:
            raise ValueError(
                "Production Cube Arena requires inference_batch_wait_ms=4.0"
            )

    def _validate_metadata(self, metadata: Mapping[str, object]) -> None:
        if metadata.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Checkpoint is not a Cube V2 training checkpoint")
        if int(metadata.get("size", -1)) != self.size:
            raise ValueError("Cube Arena checkpoint size mismatch")
        if metadata.get("architecture_id") != ARCHITECTURE_ID:
            raise ValueError("Cube Arena checkpoint architecture id mismatch")
        if metadata.get("architecture_fingerprint") != ARCHITECTURE_FINGERPRINT:
            raise ValueError("Cube Arena checkpoint architecture fingerprint mismatch")
        model_metadata = metadata.get("model_metadata")
        if not isinstance(model_metadata, Mapping):
            raise ValueError("Cube Arena checkpoint model metadata is missing")
        validate_cube_model_metadata(model_metadata)
        if int(model_metadata.get("size", -1)) != self.size:
            raise ValueError("Cube Arena checkpoint model topology mismatch")
        if metadata.get("game_fingerprint") != self.expected_game_fingerprint:
            raise ValueError("Cube Arena checkpoint game/rules fingerprint mismatch")
        if metadata.get("observation_fingerprint") != self.expected_observation_fingerprint:
            raise ValueError("Cube Arena checkpoint observation fingerprint mismatch")
        for key in (
            "target_contract_fingerprint",
            "selfplay_semantics_fingerprint",
            "model_hash",
        ):
            value = metadata.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Cube Arena checkpoint is missing {key}")

    @staticmethod
    def _payload(path: Path, sidecar: Mapping[str, object]) -> Mapping[str, object]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Cube Arena checkpoint payload schema mismatch")
        payload_metadata = payload.get("metadata")
        if not isinstance(payload_metadata, Mapping):
            raise ValueError("Cube Arena checkpoint payload metadata is missing")
        semantic_sidecar = {
            key: value for key, value in sidecar.items() if key != "checkpoint_sha256"
        }
        if dict(payload_metadata) != semantic_sidecar:
            raise ValueError("Cube Arena checkpoint payload/sidecar metadata mismatch")
        if "model_state_dict" not in payload:
            raise ValueError("Cube Arena checkpoint payload has no model state")
        return payload

    def load_identity(self, path: Path) -> CheckpointIdentity:
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Cube Arena checkpoint or metadata: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError("Cube Arena checkpoint sidecar is malformed")
        self._validate_metadata(metadata)
        artifact_sha = file_sha256(path)
        reported_sha = metadata.get("checkpoint_sha256")
        if not isinstance(reported_sha, str) or not _same_sha(artifact_sha, reported_sha):
            raise ValueError("Cube Arena checkpoint SHA-256 mismatch")
        self._payload(path, metadata)
        architecture_config = {
            "size": self.size,
            "game_fingerprint": metadata["game_fingerprint"],
            "observation_fingerprint": metadata["observation_fingerprint"],
            "architecture_id": metadata["architecture_id"],
            "architecture_fingerprint": metadata["architecture_fingerprint"],
            "target_contract_fingerprint": metadata["target_contract_fingerprint"],
            "selfplay_semantics_fingerprint": metadata["selfplay_semantics_fingerprint"],
        }
        return CheckpointIdentity(
            path=path,
            model_hash=str(metadata["model_hash"]),
            artifact_sha256=artifact_sha,
            architecture_config=architecture_config,
            metadata=dict(metadata),
        )

    def load_parent_model(
        self, identity: CheckpointIdentity, device: torch.device
    ) -> torch.nn.Module:
        payload = self._payload(identity.path, identity.metadata)
        model_metadata = identity.metadata.get("model_metadata")
        if not isinstance(model_metadata, Mapping):
            raise ValueError("Cube Arena model metadata is missing")
        model = build_cube_model_from_metadata(model_metadata)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        if cube_graphnet_v2_model_hash(model) != identity.model_hash:
            raise RuntimeError("Parent inference broker loaded the wrong Cube checkpoint")
        model.eval()
        return model

    def build_tasks(
        self,
        *,
        run_id: str,
        comparison: str,
        candidate: CheckpointIdentity,
        reference: CheckpointIdentity,
        master_seed: int,
        games: int,
        workers: int,
        workload: Mapping[str, object] | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        if candidate.architecture_config != reference.architecture_config:
            raise ValueError("Cube Arena checkpoints are scientifically incompatible")
        if type(games) is not int or games <= 0 or games % 2:
            raise ValueError("Cube Arena games must be a positive even number")
        pairs = games // 2
        startset = build_cube_arena_startset(
            size=self.size,
            master_seed=int(master_seed),
            pairs=pairs,
        )
        workload = workload or {}
        expected_startset_fingerprint = workload.get("startset_fingerprint")
        if expected_startset_fingerprint is not None and str(expected_startset_fingerprint) != startset.fingerprint:
            raise ValueError("Cube Arena workload startset fingerprint mismatch")
        self._last_startset = startset
        tasks: list[dict[str, object]] = []
        for pair_index, start in enumerate(startset.starts):
            pair_id = f"{comparison}--{start.start_id}"
            pair_seed = derive_seed(master_seed, pair_id, "cube-arena-pair")
            for suffix, candidate_black in (("g1", True), ("g2", False)):
                game_id = f"{pair_id}--{suffix}"
                tasks.append(
                    {
                        "run_id": run_id,
                        "comparison": comparison,
                        "pair_id": pair_id,
                        "game_id": game_id,
                        "start_id": start.start_id,
                        "start_kind": start.start_kind,
                        "opening_actions": list(start.opening_actions),
                        "opening_ply": start.opening_ply,
                        "start_fingerprint": start.start_fingerprint,
                        "startset_fingerprint": startset.fingerprint,
                        "candidate_black": candidate_black,
                        "candidate_hash": candidate.model_hash,
                        "reference_hash": reference.model_hash,
                        "candidate_artifact_sha256": candidate.artifact_sha256,
                        "reference_artifact_sha256": reference.artifact_sha256,
                        "game_seed": derive_seed(pair_seed, suffix),
                        "worker_id": len(tasks) % workers,
                    }
                )
        return tasks, pairs

    def worker_main(
        self,
        worker_id: int,
        task_queue: Any,
        games_per_worker: int,
        worker_local_wait_ms: float,
        candidate_hash: str,
        reference_hash: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queues: Any,
        start_event: Any,
    ) -> None:
        from tools.arena_profiles.cube_v2_worker import run_cube_v2_worker

        run_cube_v2_worker(
            size=self.size,
            search_config=self.search_config,
            worker_id=worker_id,
            task_queue=task_queue,
            games_per_worker=games_per_worker,
            worker_local_wait_ms=worker_local_wait_ms,
            candidate_hash=candidate_hash,
            reference_hash=reference_hash,
            input_slot=input_slot,
            policy_slot=policy_slot,
            wdl_slot=wdl_slot,
            request_queue=request_queue,
            response_queues=response_queues,
            start_event=start_event,
        )

    @staticmethod
    def forward_policy_wdl_logits(
        model: torch.nn.Module,
        batch: torch.Tensor,
    ) -> object:
        return model.infer_policy_wdl(batch)

    def infer_batch(
        self,
        model: torch.nn.Module,
        cpu_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inferred = infer_policy_wdl_batch(
            model,
            cpu_batch,
            device,
            observation_shape=self.observation_shape,
            policy_size=self.policy_size,
            wdl_size=self.wdl_size,
            forward_key=self.profile_id,
            forward_policy_wdl_logits=self.forward_policy_wdl_logits,
        )
        self.last_infer_timing = inferred.timing
        return inferred.policy, inferred.wdl

    def summarize(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        candidate_label: str,
        reference_label: str,
        pairs: int,
    ) -> dict[str, object]:
        candidate_wins = sum(row.get("mapped_result") == "A_WIN" for row in records)
        reference_wins = sum(row.get("mapped_result") == "B_WIN" for row in records)
        draws = sum(row.get("mapped_result") == "DRAW" for row in records)
        invalid = sum(
            str(row.get("technical_termination") or "").startswith("ERROR_")
            for row in records
        )
        technical = sum(
            row.get("technical_termination") is not None
            and not str(row.get("technical_termination")).startswith("ERROR_")
            for row in records
        )
        valid = candidate_wins + reference_wins + draws
        by_pair: dict[str, list[Mapping[str, object]]] = {}
        for row in records:
            by_pair.setdefault(str(row.get("pair_id")), []).append(row)
        return {
            "candidate_label": candidate_label,
            "reference_label": reference_label,
            "pairs": int(pairs),
            "games_requested": len(records),
            "games_valid": valid,
            "valid_games": valid,
            "candidate_wins": candidate_wins,
            "reference_wins": reference_wins,
            "draws": draws,
            "technical_games": technical,
            "technical_only_games": technical,
            "invalid_games": invalid,
            "unique_start_count": len({str(row.get("start_id")) for row in records}),
            "empty_control_pairs": len({
                str(row.get("pair_id")) for row in records
                if row.get("start_kind") == "empty_control"
            }),
            "diverse_pairs": len({
                str(row.get("pair_id")) for row in records
                if row.get("start_kind") == "diverse"
            }),
            "valid_paired_starts": sum(
                len(group) == 2
                and all(
                    row.get("mapped_result") in {"A_WIN", "B_WIN", "DRAW"}
                    for row in group
                )
                for group in by_pair.values()
            ),
            "empty_control_results": {
                "candidate_wins": sum(
                    row.get("mapped_result") == "A_WIN" and row.get("start_kind") == "empty_control"
                    for row in records
                ),
                "reference_wins": sum(
                    row.get("mapped_result") == "B_WIN" and row.get("start_kind") == "empty_control"
                    for row in records
                ),
                "draws": sum(
                    row.get("mapped_result") == "DRAW" and row.get("start_kind") == "empty_control"
                    for row in records
                ),
            },
            "completion_status": "COMPLETE",
        }

    def scientific_contract(
        self, config: ArenaExecutionConfig
    ) -> Mapping[str, object]:
        return {
            "contract": "cube-v2-arena-stage7",
            "topology": f"cube{self.size}",
            "size": self.size,
            "opening": GENERATOR_VERSION,
            "startset_schema": STARTSET_SCHEMA,
            "startset_fingerprint": (
                None if self._last_startset is None else self._last_startset.fingerprint
            ),
            "topology_fingerprint": self.topology.fingerprint,
            "game_fingerprint": self.expected_game_fingerprint,
            "observation_fingerprint": self.expected_observation_fingerprint,
            "empty_board_control_pair": True,
            "paired_starts_color_swap": True,
            "search": self.search_config.identity_payload(),
            "search_fingerprint": self.search_config.fingerprint,
            "games": int(config.games),
            "technical_outcomes_excluded_from_formal_wdl": True,
            "model_gating": False,
        }


__all__ = ["CubeV2ArenaProfile"]
