#!/usr/bin/env python3
"""Reproducible bounded Torus9 continuation audit for immutable source revisions.

All generated files must live under /tmp. The self-play workers use the
production Torus9 adapter, shared inference broker, PUCT session, target
builder, and ordinary trainer. This harness never invokes a production job or
publishes replay/checkpoint artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_DIAGNOSTICS_PATH = Path(__file__).with_name("scientific_continuation_diagnostics.py")
_DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "scientific_continuation_diagnostics", _DIAGNOSTICS_PATH)
if _DIAGNOSTICS_SPEC is None or _DIAGNOSTICS_SPEC.loader is None:
    raise RuntimeError(f"Could not load diagnostics module: {_DIAGNOSTICS_PATH}")
_DIAGNOSTICS = importlib.util.module_from_spec(_DIAGNOSTICS_SPEC)
sys.modules[_DIAGNOSTICS_SPEC.name] = _DIAGNOSTICS
_DIAGNOSTICS_SPEC.loader.exec_module(_DIAGNOSTICS)

EXPECTED_REVISIONS = {
    "A": "3010cc0de2fcb3770cdceb9abad98a66d0ff956c",
    "B": "86d72685852f7df2b7100971d16be4406252cb21",
    "C": "bbb2a93ea185195a8c0a029b091b92cd2b6dfb59",
}
EXPECTED_M233_SHA256 = "15aedb102c386837227daa660d412eab7e937f96e022dc93c1d4503675d81cfd"
FIXED_RUN_ID = "scientific-continuation-gate-fixed-v1"
PCR_RUN_ID = "scientific-continuation-gate-pcr-v1"
MASTER_SEED = 2026092901
FIXED_SIMULATIONS = 200
PCR_CONFIG = {"cheap_simulations": 100, "full_simulations": 500, "full_probability": 0.25}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tmp_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(Path("/tmp")):
        raise ValueError(f"Audit output must be under /tmp, got {resolved}")
    return resolved


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row).decode("utf-8") + "\n")


def first_mapping_difference(left: Mapping[str, Any], right: Mapping[str, Any],
                             path: str = "") -> dict[str, Any] | None:
    keys = sorted(set(left) | set(right))
    for key in keys:
        label = f"{path}.{key}" if path else key
        if key not in left or key not in right:
            return {"path": label, "left": left.get(key, "<missing>"),
                    "right": right.get(key, "<missing>")}
        a, b = left[key], right[key]
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            nested = first_mapping_difference(a, b, label)
            if nested is not None:
                return nested
        elif a != b:
            return {"path": label, "left": a, "right": b}
    return None


def _clean_import_path(repo: Path) -> None:
    task_root = Path(__file__).resolve().parents[2]
    kept = []
    for item in sys.path:
        try:
            resolved = Path(item or os.getcwd()).resolve()
        except OSError:
            kept.append(item)
            continue
        if resolved == task_root or task_root in resolved.parents:
            continue
        kept.append(item)
    sys.path[:] = kept
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def _revision(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip()


def _runtime(repo: Path, *, expected_revision: str | None = None,
             device: str = "cuda") -> dict[str, Any]:
    repo = repo.resolve()
    actual_revision = _revision(repo)
    if expected_revision and actual_revision != expected_revision:
        raise ValueError(f"Source revision drift: expected {expected_revision}, got {actual_revision}")
    if device != "cuda":
        raise ValueError("The Legion acceptance gate requires device=cuda")
    _clean_import_path(repo)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Legion acceptance gate")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    return {"torch": torch, "repo": repo, "revision": actual_revision,
            "device": device, "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda}


def _load_model(checkpoint: Path, torch: Any, device: str):
    from gocube_golden.neural import model_hash
    from gocube_golden.torus9_adaptation import AdaptationModel
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("M233 checkpoint has no metadata object")
    if metadata.get("architecture_id") != "GoldenGraphNetV2-Torus9-M137-5CH":
        raise ValueError("M233 architecture identity drift")
    if float(metadata.get("komi", -1.0)) != 1.5:
        raise ValueError("M233 komi identity drift")
    model = AdaptationModel()
    model.load_state_dict(raw["model_state_dict"], strict=True)
    found_model_hash = model_hash(model)
    if found_model_hash != metadata.get("model_hash"):
        raise ValueError("M233 model hash does not match checkpoint metadata")
    model.to(device).eval()
    return raw, model, found_model_hash


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    metadata = {"dtype": str(cpu.dtype), "shape": list(cpu.shape)}
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _json_action(action: object) -> int | str:
    if isinstance(action, bool) or not isinstance(action, (int, str)):
        raise TypeError(f"Unsupported Torus9 action identity: {action!r}")
    return action


def semantic_game_from_record(record: Mapping[str, Any], core: Any,
                              *, fixed_simulations: int = FIXED_SIMULATIONS) -> dict[str, Any]:
    from gocube_golden.provenance import derive_seed
    state = core.torus9_state_from_identity(record["start_state"])
    semantic_positions = []
    trace = [_json_action(action) for action in record["final_action_trace"]]
    positions = list(record["positions"])
    if len(positions) != len(trace):
        raise ValueError(f"Game {record.get('game_id')} has different position and trace lengths")
    for index, position in enumerate(positions):
        state = core.torus9_state_from_identity(position["state"])
        if core.torus9_state_identity(state) != position["state"]:
            raise ValueError(f"State identity is not canonical at {record['game_id']} ply {index + 1}")
        legal = core.prepare_legal_actions(state)
        actions = [_json_action(action) for action in legal.actions]
        legal_mask = [bool(value) for value in legal.action_mask]
        search_mode = str(position.get("search_mode", "fixed"))
        cap = position.get("search_simulations")
        if cap is None:
            cap = fixed_simulations
        search_seed = int(position["search_seed"])
        expected_seed = derive_seed(int(record["game_seed"]), index + 1, "search")
        if search_seed != expected_seed:
            raise ValueError(f"Search seed identity drift at {record['game_id']} ply {index + 1}")
        selected = _json_action(position["selected_action"])
        if selected != trace[index]:
            raise ValueError(f"Action trace drift at {record['game_id']} ply {index + 1}")
        semantic_positions.append({
            "ply": int(position["ply"]),
            "state_identity": position["state"],
            "state_sha256": sha256_json(position["state"]),
            "side_to_move": str(position["side_to_move"]),
            "selected_action": selected,
            "simulation_cap": int(cap),
            "search_mode": search_mode,
            "legal_action_identity": {
                "actions": actions,
                "mask_sha256": sha256_json(legal_mask),
                "legal_count": sum(legal_mask),
            },
            "root_visits": [int(value) for value in position["root_visits"]],
            "training_policy": [float(value) for value in position["pi"]],
            "training_eligible": bool(position.get("training_eligible", search_mode != "cheap")),
            "rng_identity": {"game_seed": int(record["game_seed"]),
                             "search_seed": search_seed},
        })
    final_state = core.torus9_state_from_identity(record["start_state"])
    for action in trace:
        final_state = core.apply_action(final_state, action).after
    if record.get("technical_termination") is None:
        if not final_state.is_terminal:
            raise ValueError(f"Formal game {record.get('game_id')} did not terminate")
        if trace[-2:] != [core.PASS, core.PASS]:
            raise ValueError(f"Formal game {record.get('game_id')} did not end on two passes")
        terminal_reason = "DOUBLE_PASS"
        score = core.score_terminal(final_state)
        margin_black = float(score.margin_black)
    else:
        terminal_reason = str(record["technical_termination"])
        margin_black = None
    return {
        "run_id": str(record["run_id"]),
        "game_id": str(record["game_id"]),
        "master_seed": int(record["master_seed"]),
        "game_seed": int(record["game_seed"]),
        "model_hash": str(record["model_hash"]),
        "start_state": record["start_state"],
        "action_trace": trace,
        "plies": len(trace),
        "side_to_move_trace": [position["side_to_move"] for position in semantic_positions],
        "terminal_reason": terminal_reason,
        "winner_wdl": record.get("formal_result"),
        "final_margin_black": margin_black,
        "positions": semantic_positions,
        "technical_status": {
            "termination": record.get("technical_termination"),
            "error": record.get("error"),
        },
    }


def _target_summaries(records: Sequence[Mapping[str, Any]], target_games: Sequence[Mapping[str, Any]],
                      core: Any, torch: Any) -> dict[str, Any]:
    from gocube_golden.torus9_adaptation import validate_game
    targets_by_id = {str(game["game_id"]): game for game in target_games}
    target_rows = []
    field_rows: dict[str, list[tuple[str, str]]] = {
        key: [] for key in ("observation", "legal", "visits", "pi", "z", "ownership", "score")
    }
    expected_rows = 0
    for wrapped in sorted(records, key=lambda row: str(row["semantic"]["game_id"])):
        raw = wrapped["raw_record"]
        game_id = str(raw["game_id"])
        eligible_positions = [position for position in raw["positions"]
                              if bool(position.get("training_eligible", True))]
        target = targets_by_id.get(game_id)
        if not eligible_positions:
            if target is not None:
                raise ValueError(f"All-cheap game {game_id} produced learner samples")
            continue
        if target is None:
            raise ValueError(f"Eligible game {game_id} is missing learner targets")
        validate_game(target)
        if int(target["observation"].shape[0]) != len(eligible_positions):
            raise ValueError(f"Target count does not match eligible positions for {game_id}")
        final = core.torus9_state_from_identity(raw["start_state"])
        for action in raw["final_action_trace"]:
            final = core.apply_action(final, action).after
        if not final.is_terminal:
            raise ValueError(f"Terminal targets cannot be built from incomplete game {game_id}")
        for row_index, position in enumerate(eligible_positions):
            ply = int(position["ply"])
            row_id = f"{game_id}:{ply:04d}"
            sample_fields = {}
            sample_tensors = {}
            for field_name in field_rows:
                tensor = target[field_name][row_index]
                sample_tensors[field_name] = tensor
                field_hash = _tensor_sha256(tensor, torch)
                sample_fields[field_name] = field_hash
                field_rows[field_name].append((row_id, field_hash))
            state = core.torus9_state_from_identity(position["state"])
            legal = core.prepare_legal_actions(state)
            expected_legal = torch.tensor(legal.action_mask, dtype=torch.bool)
            expected_visits = torch.tensor(position["root_visits"], dtype=torch.long)
            expected_pi = torch.tensor(position["pi"], dtype=torch.float32)
            expected_z = torch.tensor(core.torus9_z_target(raw["formal_result"], state.side_to_move),
                                      dtype=torch.float32)
            expected_ownership = torch.tensor(core.torus9_ownership_target(final, state.side_to_move),
                                              dtype=torch.long)
            expected_score = torch.tensor(core.torus9_score_target(final, state.side_to_move) / 81.5,
                                          dtype=torch.float32)
            for field_name, expected in (("legal", expected_legal), ("visits", expected_visits),
                                         ("pi", expected_pi), ("z", expected_z),
                                         ("ownership", expected_ownership), ("score", expected_score)):
                if not torch.equal(sample_tensors[field_name], expected):
                    raise ValueError(f"{field_name} target mismatch at {row_id}")
            sample_identity = {
                "row_id": row_id,
                "game_id": game_id,
                "ply": ply,
                "selected_action": position["selected_action"],
                "terminal_winner": raw["formal_result"],
                "fields": sample_fields,
            }
            sample_identity["semantic_sha256"] = sha256_json(sample_identity)
            target_rows.append(sample_identity)
            expected_rows += 1
    extra_games = sorted(set(targets_by_id) - {str(row["raw_record"]["game_id"]) for row in records})
    if extra_games:
        raise ValueError(f"Target builder emitted unknown games: {extra_games[:5]}")
    field_digests = {
        name: sha256_json([{"row_id": row_id, "sha256": digest} for row_id, digest in values])
        for name, values in field_rows.items()
    }
    return {
        "sample_count": expected_rows,
        "samples": target_rows,
        "field_sha256": field_digests,
        "sample_identity_sha256": sha256_json(target_rows),
    }


def _trace_observation(request: Any, torch: Any) -> tuple[str, str]:
    """Materialize the exact Torus9 input through the production observation writer."""
    from gocube_golden.torus9_adaptation import write_observation
    observation = torch.empty((5, 81), dtype=torch.float32)
    write_observation((request.state, request.legal_context), observation)
    raw = observation.contiguous().view(torch.uint8).numpy().tobytes()
    return _tensor_sha256(observation, torch), raw.hex()


def _install_tracing_session(owner_module: Any, core: Any, torch: Any):
    """Build an audit-only Session subclass over the source revision's real PUCT."""
    from gocube_golden.search import SearchEvaluationRequest

    base = getattr(owner_module, "_continuation_audit_session_base",
                   owner_module.SequentialPUCTSession)
    owner_module._continuation_audit_session_base = base

    class TracingSequentialPUCTSession(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._audit_trace: dict[str, Any] | None = None
            self._audit_metadata: dict[str, Any] = {}
            self._audit_request_count = 0
            self._audit_next_simulation = 0
            self._audit_active_simulation: dict[str, Any] | None = None
            self._audit_root_node = None
            self._audit_root_eval_seen = False

        def _audit_enabled(self) -> bool:
            return self._audit_trace is not None

        def _audit_action(self, action: object) -> int | str:
            return _json_action(action)

        def _audit_state(self, state: Any) -> dict[str, Any]:
            identity = core.torus9_state_identity(state)
            return {"identity": identity, "sha256": sha256_json(identity)}

        def _audit_root_visits(self) -> list[int]:
            if self._audit_root_node is None:
                return []
            root = self._audit_root_node
            actions = self.adapter.action_space(root.state)
            return [int(root.edges[action].visits) if action in root.edges else 0
                    for action in actions]

        def _audit_root_priors(self) -> list[float]:
            if self._audit_root_node is None:
                return []
            root = self._audit_root_node
            actions = self.adapter.action_space(root.state)
            return [float(root.edges[action].prior) if action in root.edges else 0.0
                    for action in actions]

        def _audit_eval_request(self, evaluation: Any) -> dict[str, Any]:
            request = self.pending
            if not isinstance(request, SearchEvaluationRequest):
                raise RuntimeError("tracing session resumed without a pending Evaluation request")
            observation_sha, observation_blob = _trace_observation(request, torch)
            state = self._audit_state(request.state)
            legal_mask = [bool(value) for value in request.legal_context.action_mask]
            policy = [float(value) for value in evaluation.policy]
            wdl = [float(value) for value in evaluation.wdl]
            policy_tensor = torch.tensor(policy, dtype=torch.float32)
            wdl_tensor = torch.tensor(wdl, dtype=torch.float32)
            row = {
                "session_request_index": self._audit_request_count,
                "state_identity": state["identity"], "state_sha256": state["sha256"],
                "legal_mask_sha256": sha256_json(legal_mask), "legal_mask": legal_mask,
                "legal_actions": [_json_action(action) for action in request.legal_context.actions],
                "observation_sha256": observation_sha,
                "observation_dtype": "float32", "observation_shape": [5, 81],
                "observation_blob_hex": observation_blob,
                "policy": policy, "wdl": wdl,
                "policy_sha256": _tensor_sha256(policy_tensor, torch),
                "wdl_sha256": _tensor_sha256(wdl_tensor, torch),
            }
            self._audit_request_count += 1
            self._audit_trace["evaluation_requests"].append(row)
            if self._audit_active_simulation is not None:
                self._audit_active_simulation["evaluation_request_index"] = row[
                    "session_request_index"]
            return row

        def resume(self, evaluation: Any):
            if not self._audit_enabled():
                return super().resume(evaluation)
            row = self._audit_eval_request(evaluation)
            is_root = not self._audit_root_eval_seen
            if is_root:
                self._audit_trace["root_evaluation_request_index"] = row[
                    "session_request_index"]
            step = super().resume(evaluation)
            if is_root:
                self._audit_root_eval_seen = True
                self._audit_trace["root_priors"] = self._audit_root_priors()
                self._audit_trace["root_priors_sha256"] = sha256_json(
                    self._audit_trace["root_priors"])
            return step

        def _traverse(self, root: Any):
            if not self._audit_enabled():
                return super()._traverse(root)
            self._audit_root_node = root
            event: dict[str, Any] = {
                "simulation_index": self._audit_next_simulation,
                "puct_decisions": [],
            }
            self._audit_root_node = root
            event["root_visits_before"] = self._audit_root_visits()
            self._audit_active_simulation = event
            path, leaf = super()._traverse(root)
            event["traversal_path"] = [
                {"state_sha256": self._audit_state(parent.state)["sha256"],
                 "action": self._audit_action(action)}
                for parent, action, _edge in path]
            event["leaf"] = self._audit_state(leaf.state)
            event["leaf_is_terminal"] = bool(self.adapter.is_terminal(leaf.state))
            self._audit_trace["simulations"].append(event)
            return path, leaf

        def _select(self, node: Any):
            if not self._audit_enabled():
                return super()._select(node)
            rng_before = sha256_json(repr(self._rng.getstate()))
            selected_action, selected_edge = super()._select(node)
            total_visits = sum(edge.visits for edge in node.edges.values())
            scale = math.sqrt(total_visits + 1.0)
            candidates = []
            for action, edge in node.edges.items():
                q = edge.q if edge.visits else float(self.settings.fpu)
                u = float(self.settings.cpuct) * edge.prior * scale / (1.0 + edge.visits)
                score = q + u
                candidates.append({
                    "action": self._audit_action(action),
                    "action_index": int(self.adapter.action_index(node.state, action)),
                    "visits": int(edge.visits), "q": float(q),
                    "q_value_sum": float(edge.value_sum), "prior": float(edge.prior),
                    "u": float(u), "score": float(score),
                })
            candidates.sort(key=lambda item: (-item["score"], item["action_index"]))
            node_identity = self._audit_state(node.state)
            root_identity = self._audit_trace["root_identity"]["state_sha256"]
            if node_identity["sha256"] == root_identity:
                retained = candidates
            else:
                retained = candidates[:2]
            if self._audit_active_simulation is not None:
                event = self._audit_active_simulation
                depth = len(event["puct_decisions"])
                selected_index = int(self.adapter.action_index(node.state, selected_action))
                event["puct_decisions"].append({
                    "depth": depth, "node_state_sha256": node_identity["sha256"],
                    "node_is_root": node_identity["sha256"] == root_identity,
                    "rng_state_sha256_before": rng_before,
                    "candidate_scores": retained,
                    "selected_action": self._audit_action(selected_action),
                    "selected_action_index": selected_index,
                    "selected_edge_visits_before": int(selected_edge.visits),
                    "selected_edge_q_before": float(
                        selected_edge.q if selected_edge.visits else float(self.settings.fpu)),
                    "selected_edge_prior": float(selected_edge.prior),
                })
            return selected_action, selected_edge

        def _backup(self, path: Sequence[Any], utility: float) -> float:
            if not self._audit_enabled():
                return super()._backup(path, utility)
            result = super()._backup(path, utility)
            event = self._audit_active_simulation
            if event is not None:
                event["leaf_utility"] = float(utility)
                event["backup_utility"] = float(result)
                event["root_visits_after"] = self._audit_root_visits()
                event["selected_root_edge"] = (
                    self._audit_action(path[0][1]) if path else None)
                self._audit_next_simulation += 1
                self._audit_active_simulation = None
            return result

    return TracingSequentialPUCTSession


class _RootAuditFactory:
    """Picklable test-only wrapper for root contract and optional search trace."""

    def __init__(self, output: Path, *, trace_game_id: str | None = None,
                 trace_ply: int = 1, observation_queue: Any = None) -> None:
        self.output = output
        self.trace_game_id = trace_game_id
        self.trace_ply = int(trace_ply)
        self.observation_queue = observation_queue

    def __call__(self, context: object, game_id: str, client: object):
        from gocube_golden import torus9_selfplay as owner_module
        from gocube_golden import torus9_monolith as core
        from gocube_golden.torus9_selfplay import _make_torus9_cooperative_game
        game = _make_torus9_cooperative_game(context, game_id, client)
        trace_target = str(game_id) == self.trace_game_id
        if trace_target:
            owner_module.SequentialPUCTSession = _install_tracing_session(
                owner_module, core, core.torch)
        original_start = game._start_search

        def observed_start() -> None:
            original_start()
            noise = getattr(game, "_root_noise", None)
            item = {
                "game_id": game.game_id,
                "ply": len(game.trace) + 1,
                "game_seed": int(game.game_seed),
                "search_seed": int(__import__("gocube_golden.provenance", fromlist=["derive_seed"])
                                    .derive_seed(game.game_seed, len(game.trace) + 1, "search")),
                "search_mode": getattr(game, "_search_mode", "fixed"),
                "simulation_cap": int(getattr(
                    game, "_search_simulations", game.context.contract.simulations)),
                "root_noise_enabled": noise is not None,
                "root_noise_alpha": None if noise is None else float(noise.alpha),
                "root_noise_epsilon": None if noise is None else float(game.context.contract.dirichlet_epsilon),
                "pid": os.getpid(),
            }
            path = self.output / f"root-audit-{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(canonical_json(item).decode("utf-8") + "\n")

            if trace_target and int(item["ply"]) == self.trace_ply:
                session = game._session
                root_identity = core.torus9_state_identity(game.state)
                session._audit_metadata = {
                    "game_id": game.game_id, "ply": self.trace_ply,
                    "worker_id": int(getattr(client, "worker_id", -1)),
                    "lane_id": int(getattr(client, "lane_id", -1)),
                    "pid": os.getpid(), "game_seed": int(game.game_seed),
                    "search_seed": int(item["search_seed"]),
                    "root_identity": root_identity,
                    "root_state_sha256": sha256_json(root_identity),
                    "simulation_cap": int(item["simulation_cap"]),
                    "settings": {name: getattr(session.settings, name) for name in
                                 ("simulations", "cpuct", "fpu", "deterministic_tie_break")
                                 if hasattr(session.settings, name)},
                    "noise": {"enabled": noise is not None,
                              "alpha": None if noise is None else float(noise.alpha),
                              "epsilon": None if noise is None else float(
                                  game.context.contract.dirichlet_epsilon),
                              "seed": None if noise is None else int(
                                  __import__("gocube_golden.provenance", fromlist=["derive_seed"])
                                  .derive_seed(int(item["search_seed"]), "dirichlet"))},
                }
                session._audit_trace = {
                    "schema": _DIAGNOSTICS.ROOT_TRACE_SCHEMA,
                    "game_id": game.game_id, "ply": self.trace_ply,
                    "worker_id": session._audit_metadata["worker_id"],
                    "lane_id": session._audit_metadata["lane_id"],
                    "pid": os.getpid(), "game_seed": int(game.game_seed),
                    "search_seed": int(item["search_seed"]),
                    "simulation_cap": int(item["simulation_cap"]),
                    "root_identity": {"identity": root_identity,
                                      "state_sha256": session._audit_metadata[
                                          "root_state_sha256"]},
                    "settings": session._audit_metadata["settings"],
                    "noise": session._audit_metadata["noise"],
                    "evaluation_requests": [], "simulations": [],
                }
                game._audit_trace_session = session
                game._audit_trace_written = False

        game._start_search = observed_start

        if trace_target and self.observation_queue is not None:
            original_advance = game.advance

            def observed_advance():
                outcome = original_advance()
                session = getattr(game, "_audit_trace_session", None)
                if (session is not None and int(len(game.trace) + 1) == self.trace_ply
                        and getattr(outcome, "payload", None) is not None):
                    state, legal_context = outcome.payload
                    class RequestView:
                        pass
                    view = RequestView()
                    view.state, view.legal_context = state, legal_context
                    observation_sha, _blob = _trace_observation(view, core.torch)
                    payload_identities[id(outcome.payload)] = {
                        "game_id": game.game_id, "ply": self.trace_ply,
                        "session_request_index": len(session._audit_trace[
                            "evaluation_requests"]),
                        "observation_sha256": observation_sha,
                    }
                session = getattr(game, "_audit_trace_session", None)
                if (session is not None and session.result is not None
                        and not game._audit_trace_written):
                    result = session.result
                    trace = session._audit_trace
                    trace.update({
                        "root_visits": [int(value) for value in result.root_visits],
                        "pi": [float(value) for value in result.pi],
                        "selected_action": _json_action(result.action),
                        "simulations_completed": len(trace["simulations"]),
                    })
                    trace_path = self.output / (
                        f"root-trace-{os.getpid()}-{self.trace_ply:04d}.json")
                    write_json(trace_path, trace)
                    stream_path = self.output / (
                        f"evaluation-stream-{os.getpid()}-{self.trace_ply:04d}.json")
                    write_json(stream_path,
                               _DIAGNOSTICS.serialize_evaluation_stream(trace))
                    game._audit_trace_written = True
                return outcome

            game.advance = observed_advance
            payload_identities = getattr(client, "_continuation_audit_payloads", None)
            if payload_identities is None:
                payload_identities = {}
                client._continuation_audit_payloads = payload_identities
            if not hasattr(client, "_continuation_audit_original_request_shared_batch"):
                original_request_shared_batch = client.request_shared_batch
                client._continuation_audit_original_request_shared_batch = (
                    original_request_shared_batch)

                def observed_request_shared_batch(rows: Sequence[tuple[int, object]]):
                    request_id = int(client._next_id)
                    for slot, payload in rows:
                        identity = payload_identities.pop(id(payload), None)
                        if identity is not None:
                            self.observation_queue.put({
                                **identity, "worker_id": int(client.worker_id),
                                "lane_id": int(client.lane_id), "request_id": request_id,
                                "slot_id": int(slot),
                            })
                    return original_request_shared_batch(rows)

                client.request_shared_batch = observed_request_shared_batch
        return game


class _InstrumentedAdapter:
    def __init__(self, delegate: object, audit_output: Path, *,
                 trace_game_id: str | None = None, trace_ply: int = 1,
                 observation_queue: Any = None) -> None:
        self.delegate = delegate
        self.audit_output = audit_output
        self.trace_game_id = trace_game_id
        self.trace_ply = int(trace_ply)
        self.observation_queue = observation_queue

    @property
    def worker_context(self):
        return self.delegate.worker_context

    @property
    def shared_memory(self):
        return self.delegate.shared_memory

    @property
    def infer_shared_batch(self):
        return self.delegate.infer_shared_batch

    @property
    def worker_game_factory(self):
        return _RootAuditFactory(self.audit_output, trace_game_id=self.trace_game_id,
                                 trace_ply=self.trace_ply,
                                 observation_queue=self.observation_queue)

    @property
    def record_metrics(self):
        return self.delegate.record_metrics


def _run_games(*, runtime: Mapping[str, Any], model: Any, checkpoint: Path,
               output: Path, game_ids: Sequence[str], run_id: str, seed: int,
               search_mode: str, pcr: Mapping[str, Any] | None,
               workers: int, active_games_per_worker: int,
               trace_game_id: str | None = None, trace_ply: int = 1,
               progress_interval: int = 4):
    import torch
    from gocube_golden.orchestrator_v2.execution_permit import _test_authority
    from gocube_golden.selfplay_engine import SelfPlayEngineConfig, run_cooperative_selfplay
    from gocube_golden.torus9_adaptation import AdaptationSelfPlayAdapter

    simulations = int(pcr["full_simulations"]) if pcr else FIXED_SIMULATIONS
    adapter_kwargs = {"checkpoint": checkpoint, "run_id": run_id, "seed": seed,
                      "device": "cuda", "simulations": simulations}
    if search_mode == "pcr":
        adapter_kwargs.update(search_mode="pcr", pcr=dict(pcr or {}))
    delegate = AdaptationSelfPlayAdapter(model, **adapter_kwargs)
    observation_queue = None
    manager = None
    if trace_game_id is not None:
        manager = multiprocessing.Manager()
        observation_queue = manager.Queue()
    adapter = _InstrumentedAdapter(
        delegate, output, trace_game_id=trace_game_id, trace_ply=trace_ply,
        observation_queue=observation_queue)
    config = SelfPlayEngineConfig(
        workers=workers, inference_batch_cap=64, inference_batch_wait_ms=1.0,
        device="cuda", process_start_method="spawn", lanes_per_worker=1,
        active_games_per_worker=active_games_per_worker,
    )
    progress = {"completed": 0, "last_report": 0}

    def mark(done: int, total: int) -> None:
        progress["completed"] = done
        if done == total or done - progress["last_report"] >= progress_interval:
            progress["last_report"] = done
            print(f"[{runtime['revision'][:8]}] {run_id}: games {done}/{total}", flush=True)

    original_dispatch = None
    if trace_game_id is not None:
        import selfplay_engine as execution_module
        original_dispatch = execution_module._CentralInference._dispatch
        _install_batch_observer(execution_module, output, observation_queue, torch)
    try:
        with _test_authority():
            result = run_cooperative_selfplay(
                game_ids, adapter=adapter, engine_config=config,
                active_games_per_worker=active_games_per_worker,
                progress_callback=mark,
            )
    finally:
        if original_dispatch is not None:
            execution_module._CentralInference._dispatch = original_dispatch
        if manager is not None:
            manager.shutdown()
    torch.cuda.synchronize()
    return result


def _install_batch_observer(execution_module: Any, output: Path,
                            observation_queue: Any, torch: Any) -> None:
    """Capture only rows whose exact observation was requested by the traced root."""
    from queue import Empty
    original_dispatch = execution_module._CentralInference._dispatch
    candidates: dict[tuple[int, int, int], dict[str, Any]] = {}
    batch_counter = {"value": 0}

    def observed_dispatch(owner: Any, batch: Sequence[Any]) -> None:
        while True:
            try:
                request = observation_queue.get_nowait()
            except Empty:
                break
            key = (int(request["worker_id"]), int(request["request_id"]),
                   int(request["slot_id"]))
            candidates[key] = dict(request)
        rows = []
        if candidates:
            row_index = 0
            for envelope in batch:
                source = owner.shared_inputs[envelope.worker_id]
                for slot_id in envelope.slot_ids:
                    key = (int(envelope.worker_id), int(envelope.request_id), int(slot_id))
                    identity = candidates.pop(key, None)
                    if identity is not None:
                        input_row = source[slot_id]
                        observation_sha = _tensor_sha256(input_row, torch)
                        if observation_sha != identity["observation_sha256"]:
                            raise RuntimeError(
                                "traced request observation did not map to its shared inference row")
                        rows.append((row_index, envelope, int(slot_id), observation_sha,
                                     identity))
                    row_index += 1
        original_dispatch(owner, batch)
        if rows:
            batch_size = sum(int(envelope.rows) for envelope in batch)
            path = output / "inference-batch-matches.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                for row_index, envelope, slot_id, observation_sha, identity in rows:
                    policy = owner.shared_policy[envelope.worker_id][slot_id].detach().clone()
                    wdl = owner.shared_wdl[envelope.worker_id][slot_id].detach().clone()
                    item = {
                        "batch_index": batch_counter["value"], "batch_size": batch_size,
                        "row_index": row_index,
                        "worker_id": int(envelope.worker_id), "lane_id": int(envelope.lane_id),
                        "request_id": int(envelope.request_id), "slot_id": slot_id,
                        "trace_request": identity,
                        "observation_sha256": observation_sha,
                        "policy": [float(value) for value in policy.tolist()],
                        "wdl": [float(value) for value in wdl.tolist()],
                        "policy_sha256": _tensor_sha256(policy, torch),
                        "wdl_sha256": _tensor_sha256(wdl, torch),
                    }
                    stream.write(canonical_json(item).decode("utf-8") + "\n")
        batch_counter["value"] += 1

    execution_module._CentralInference._dispatch = observed_dispatch


def _read_root_audit(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.glob("root-audit-*.jsonl")):
        rows.extend(read_jsonl(path))
    return sorted(rows, key=lambda row: (str(row["game_id"]), int(row["ply"])))


def validate_root_audit(records: Sequence[Mapping[str, Any]], roots: Sequence[Mapping[str, Any]],
                        *, mode: str, fixed_simulations: int = FIXED_SIMULATIONS) -> dict[str, Any]:
    expected: dict[tuple[str, int], Mapping[str, Any]] = {}
    for wrapped in records:
        raw = wrapped["raw_record"]
        for position in raw["positions"]:
            expected[(str(raw["game_id"]), int(position["ply"]))] = position
    actual: dict[tuple[str, int], Mapping[str, Any]] = {}
    for root in roots:
        key = (str(root["game_id"]), int(root["ply"]))
        if key in actual:
            raise ValueError(f"Duplicate observed search root {key}")
        actual[key] = root
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise ValueError(f"Root audit and game records disagree: missing={missing}, extra={extra}")
    invalid = []
    for key, position in expected.items():
        root = actual[key]
        recorded_mode = str(position.get("search_mode", "fixed"))
        cap = int(position.get("search_simulations") or fixed_simulations)
        should_be_eligible = recorded_mode != "cheap"
        should_have_noise = recorded_mode != "cheap"
        if mode == "fixed":
            recorded_mode, cap, should_be_eligible, should_have_noise = "fixed", fixed_simulations, True, True
        if (root["search_mode"] != recorded_mode or int(root["simulation_cap"]) != cap
                or bool(root["root_noise_enabled"]) != should_have_noise
                or (root["root_noise_enabled"] and float(root["root_noise_alpha"]) != 0.11)
                or (root["root_noise_enabled"] and float(root["root_noise_epsilon"]) != 0.25)
                or bool(position.get("training_eligible", recorded_mode != "cheap")) != should_be_eligible
                or sum(int(value) for value in position["root_visits"]) != cap):
            invalid.append({"game_id": key[0], "ply": key[1], "root": dict(root),
                            "position": {"mode": recorded_mode, "cap": cap,
                                         "training_eligible": position.get("training_eligible", True),
                                         "root_visits_sum": sum(position["root_visits"])}})
    if invalid:
        raise ValueError(f"Invalid PCR/fixed root contract: {invalid[0]}")
    return {"roots": len(roots), "cheap_roots": sum(r["search_mode"] == "cheap" for r in roots),
            "full_roots": sum(r["search_mode"] == "full" for r in roots),
            "fixed_roots": sum(r["search_mode"] == "fixed" for r in roots),
            "invalid_cap_noise_eligibility_combinations": 0}


def _target_dataset(records: Sequence[Any], raw_wrapped: Sequence[Mapping[str, Any]],
                    output: Path, torch: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from gocube_golden.torus9_adaptation import FINGERPRINT, game_targets, save_torch
    from gocube_golden import torus9_monolith as core
    targets = []
    for record in records:
        target = game_targets(record)
        if target is not None:
            target["split"] = "train"
            positions = [position for position in record.positions
                         if getattr(position, "training_eligible", True)]
            target["sample_ids"] = [f"{record.game_id}:{position.ply:04d}" for position in positions]
            targets.append(target)
    summary = _target_summaries(raw_wrapped, targets, core, torch)
    dataset_path = output / "dataset.pt"
    save_torch(dataset_path, {"contract": FINGERPRINT,
                              "actor_hash": records[0].model_hash if records else "",
                              "games": targets})
    write_json(output / "targets.json", summary)
    return targets, summary


def _record_from_dict(raw: Mapping[str, Any], core: Any):
    from dataclasses import fields as dataclass_fields
    position_names = {field.name for field in dataclass_fields(core.Torus9SelfPlayPosition)}
    record_names = {field.name for field in dataclass_fields(core.Torus9SelfPlayGameRecord)}
    positions = []
    for item in raw["positions"]:
        value = {key: entry for key, entry in item.items() if key in position_names}
        value["root_visits"] = tuple(int(v) for v in item["root_visits"])
        value["pi"] = tuple(float(v) for v in item["pi"])
        positions.append(core.Torus9SelfPlayPosition(**value))
    game = {key: value for key, value in raw.items() if key in record_names}
    game["positions"] = tuple(positions)
    game["final_action_trace"] = tuple(game["final_action_trace"])
    return core.Torus9SelfPlayGameRecord(**game)


def worker_targets(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(Path(args.repo).resolve(), expected_revision=args.expected_revision,
                       device="cuda")
    torch = runtime["torch"]
    from gocube_golden import torus9_monolith as core
    source_rows = read_jsonl(Path(args.games_jsonl))
    records = []
    wrapped = []
    for source_row in source_rows:
        raw = source_row["raw_record"]
        semantic = semantic_game_from_record(raw, core)
        digest = sha256_json(semantic)
        if digest != source_row["semantic_sha256"]:
            raise ValueError(f"Saved raw semantic hash drift for {source_row['game_id']}")
        records.append(_record_from_dict(raw, core))
        wrapped.append({"game_id": raw["game_id"], "raw_record": raw,
                        "semantic": semantic, "semantic_sha256": digest})
    _targets, target_summary = _target_dataset(records, wrapped, output, torch)
    summary = {"source_revision": runtime["revision"], "games": len(records),
               "learner_samples": target_summary["sample_count"],
               "field_sha256": target_summary["field_sha256"],
               "sample_identity_sha256": target_summary["sample_identity_sha256"]}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return summary


def worker_selfplay(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Worker output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    runtime = _runtime(repo, expected_revision=args.expected_revision, device=args.device)
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != args.expected_checkpoint_sha:
        raise ValueError(f"M233 checkpoint SHA mismatch: {checkpoint_before}")
    _raw_checkpoint, model, model_hash_value = _load_model(checkpoint, torch, args.device)
    game_ids = [f"{args.run_id}-game-{i:04d}" for i in range(args.games)]
    search_mode = "pcr" if mode == "pcr" else "fixed"
    pcr = PCR_CONFIG if mode == "pcr" else None
    result = _run_games(
        runtime=runtime, model=model, checkpoint=checkpoint, output=output,
        game_ids=game_ids, run_id=args.run_id, seed=args.seed,
        search_mode=search_mode, pcr=pcr, workers=args.workers,
        active_games_per_worker=args.active_games_per_worker,
        trace_game_id=args.trace_game_id, trace_ply=args.trace_ply,
    )
    trace_batch_rows = _enrich_root_traces(output) if args.trace_game_id is not None else 0
    records = sorted(result.records, key=lambda record: record.game_id)
    raw_rows = []
    for record in records:
        raw = record.to_dict()
        semantic = semantic_game_from_record(raw, __import__("gocube_golden.torus9_monolith", fromlist=["*"]))
        raw_rows.append({"game_id": record.game_id, "semantic": semantic,
                         "semantic_sha256": sha256_json(semantic), "raw_record": raw})
    write_jsonl(output / "games.jsonl", raw_rows)
    roots = _read_root_audit(output)
    root_summary = validate_root_audit(raw_rows, roots, mode=mode)
    technical = [row["game_id"] for row in raw_rows
                 if row["raw_record"].get("technical_termination") is not None]
    if technical:
        raise RuntimeError(f"Technical games in audit workload: {technical[:5]}")
    targets, target_summary = _target_dataset(records, raw_rows, output, torch)
    resume_shard = None
    pcr_telemetry = None
    if mode == "pcr" and args.save_shard:
        from gocube_golden.torus9_pcr import position_telemetry
        from gocube_golden.torus9_pcr import PlayoutCapRandomization
        checkpoint_ref = json.loads(Path(args.checkpoint_ref).read_text(encoding="utf-8"))["checkpoint"]
        production_targets = []
        from gocube_golden.torus9_adaptation import game_targets
        for record in records:
            target = game_targets(record)
            if target is not None:
                target["split"] = "train"
                production_targets.append(target)
        pcr_telemetry = position_telemetry(records, PlayoutCapRandomization(**PCR_CONFIG))
        resume_shard = _write_pcr_resume_shard(
            output, records, result, production_targets, pcr_telemetry, checkpoint_ref)
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Read-only M233 checkpoint changed during self-play")
    inference_batch_rows = [int(value) for value in result.telemetry.get("batch_rows", [])]
    inference_batch_distribution = {
        str(size): int(count) for size, count in sorted(Counter(inference_batch_rows).items())}
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": checkpoint_before,
        "model_hash": model_hash_value, "torch": torch.__version__, "cuda": runtime["cuda"],
        "gpu": runtime["gpu"], "device": args.device, "run_id": args.run_id,
        "master_seed": args.seed, "mode": mode, "games": len(raw_rows),
        "semantic_games_sha256": sha256_json([
            {"game_id": row["game_id"], "sha256": row["semantic_sha256"]} for row in raw_rows]),
        "plies": sum(row["semantic"]["plies"] for row in raw_rows),
        "technical_games": len(technical), "root_contract": root_summary,
        "learner_samples": target_summary["sample_count"],
        "target_field_sha256": target_summary["field_sha256"],
        "training_eligible_positions": target_summary["sample_count"],
        "raw_positions": sum(len(row["semantic"]["positions"]) for row in raw_rows),
        "checkpoint_unchanged": checkpoint_after == checkpoint_before,
        "execution": {"workers": args.workers,
                      "active_games_per_worker": args.active_games_per_worker,
                      "inference_batch_cap": 64, "inference_batch_wait_ms": 1.0,
                      "start_method": "spawn"},
        "inference_batches": {
            "forwards": int(result.telemetry.get("inference_forwards", 0)),
            "rows": int(result.telemetry.get("inference_rows", 0)),
            "batch_size_distribution": inference_batch_distribution,
            "all_batches_size_one": bool(inference_batch_rows)
                                     and set(inference_batch_rows) == {1},
        },
        "root_trace": ({"game_id": args.trace_game_id, "ply": args.trace_ply,
                        "trace_files": [str(path) for path in sorted(
                            output.glob("root-trace-*.json"))],
                        "inference_batch_match_rows": trace_batch_rows}
                       if args.trace_game_id is not None else None),
        "pcr": PCR_CONFIG if mode == "pcr" else None,
        "resume_shard": resume_shard,
        "pcr_telemetry": pcr_telemetry,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return summary


def _enrich_root_traces(output: Path) -> int:
    """Join worker Evaluation records to the exact parent dispatch request/slot."""
    match_path = output / "inference-batch-matches.jsonl"
    matches = read_jsonl(match_path) if match_path.exists() else []
    by_index: dict[int, dict[str, Any]] = {}
    for match in matches:
        identity = match["trace_request"]
        index = int(identity["session_request_index"])
        if index in by_index:
            raise ValueError(f"duplicate central batch mapping for traced request {index}")
        by_index[index] = match
    joined = 0
    for trace_path in sorted(output.glob("root-trace-*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        requests = trace["evaluation_requests"]
        if set(by_index) != set(range(len(requests))):
            raise ValueError("root trace and central batch mapping do not have one row per Evaluation")
        for request in requests:
            index = int(request["session_request_index"])
            row = by_index[index]
            mapped = row["trace_request"]
            if (mapped["game_id"] != trace["game_id"]
                    or int(mapped["ply"]) != int(trace["ply"])
                    or mapped["observation_sha256"] != request["observation_sha256"]
                    or row["observation_sha256"] != request["observation_sha256"]
                    or row["policy_sha256"] != request["policy_sha256"]
                    or row["wdl_sha256"] != request["wdl_sha256"]):
                raise ValueError(f"inference row mapping/evaluation mismatch at request {index}")
            request["inference_batch"] = {
                "batch_index": int(row["batch_index"]),
                "batch_size": int(row["batch_size"]),
                "row_index": int(row["row_index"]),
                "worker_id": int(row["worker_id"]),
                "lane_id": int(row["lane_id"]),
                "request_id": int(row["request_id"]),
                "slot_id": int(row["slot_id"]),
            }
            joined += 1
        trace["central_batch_mapping"] = {
            "exact_request_slot_mapping": True,
            "matched_rows": len(requests),
            "dispatch_rows_sha256": sha256_json([
                {"request_index": int(row["trace_request"]["session_request_index"]),
                 "batch_index": int(row["batch_index"]),
                 "batch_size": int(row["batch_size"]),
                 "row_index": int(row["row_index"]),
                 "policy_sha256": row["policy_sha256"], "wdl_sha256": row["wdl_sha256"]}
                for row in sorted(matches,
                                  key=lambda item: int(item["trace_request"][
                                      "session_request_index"]))])
        }
        write_json(trace_path, trace)
        stream_path = output / f"evaluation-stream-{trace_path.stem.removeprefix('root-trace-')}.json"
        write_json(stream_path, _DIAGNOSTICS.serialize_evaluation_stream(trace))
    return joined


def _position_examples(games_jsonl: Path, core: Any) -> list[dict[str, Any]]:
    rows = read_jsonl(games_jsonl)
    if not rows:
        raise ValueError("No fixed self-play records are available for inference integration")
    ordered = sorted(rows, key=lambda row: str(row["game_id"]))
    selections: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    def add(label: str, wrapped: Mapping[str, Any], index: int) -> None:
        raw = wrapped["raw_record"]
        position = raw["positions"][index]
        selections.append((label, wrapped, position))

    first = ordered[0]
    add("opening", first, 0)
    if len(ordered[0]["raw_record"]["positions"]) > 8:
        add("early", ordered[0], 7)
    middle_game = ordered[min(1, len(ordered) - 1)]
    middle_positions = middle_game["raw_record"]["positions"]
    add("middle", middle_game, min(len(middle_positions) - 1, max(0, len(middle_positions) // 2)))
    late_game = ordered[min(2, len(ordered) - 1)]
    late_positions = late_game["raw_record"]["positions"]
    add("late", late_game, max(0, len(late_positions) - 1))

    fewest = None
    for wrapped in ordered:
        for index, position in enumerate(wrapped["raw_record"]["positions"]):
            state = core.torus9_state_from_identity(position["state"])
            count = sum(bool(value) for value in core.prepare_legal_actions(state).action_mask)
            candidate = (count, str(wrapped["game_id"]), int(position["ply"]), wrapped, index)
            if fewest is None or candidate[:3] < fewest[:3]:
                fewest = candidate
    if fewest is not None:
        add("small_legal_set", fewest[3], fewest[4])
    else:
        add("small_legal_set", late_game, max(0, len(late_positions) - 1))

    unique: dict[str, dict[str, Any]] = {}
    for label, wrapped, position in selections:
        identity = f"{wrapped['game_id']}:{int(position['ply']):04d}"
        item = {"identity": identity, "labels": [], "raw_record": wrapped["raw_record"],
                "position": position}
        if identity in unique:
            unique[identity]["labels"].append(label)
        else:
            item["labels"] = [label]
            unique[identity] = item
    return list(unique.values())


def _direct_adapter(model: Any, checkpoint: Path, run_id: str, seed: int):
    from gocube_golden.torus9_adaptation import AdaptationSelfPlayAdapter
    return AdaptationSelfPlayAdapter(model, checkpoint=checkpoint, run_id=run_id,
                                     seed=seed, device="cuda", simulations=FIXED_SIMULATIONS)


def _masked_policy(policy: Any, legal_mask: Any, torch: Any):
    mask = torch.tensor(legal_mask, dtype=torch.bool)
    selected = policy.detach().cpu()[mask]
    total = selected.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("MCTS legal masking produced no policy mass")
    result = torch.zeros_like(policy.detach().cpu())
    result[mask] = selected / total
    return result


def _mcts_via_shared_dispatch(*, state: Any, example: Mapping[str, Any], request_id_base: int,
                              adapter: Any,
                              broker: Any, shared_input: Any, shared_policy: Any,
                              shared_wdl: Any, response_queue: Any, torch: Any,
                              core: Any, contract: Any, search_module: Any) -> dict[str, Any]:
    from gocube_golden.provenance import derive_seed
    from gocube_golden.selfplay_policy import sample_action_from_search_result
    from gocube_golden import search as search_module
    from gocube_golden.search import Evaluation, SearchEvaluationRequest, SearchResult
    from selfplay_engine import _Request

    position = example["position"]
    raw = example["raw_record"]
    search_seed = int(position["search_seed"])
    noise = core.Torus9RootNoiseEvaluator(
        None, state, seed=derive_seed(search_seed, "dirichlet"), alpha=0.11)
    session = search_module.SequentialPUCTSession(
        state, contract.settings, adapter=core.GoldenSearchAdapter(),
        seed=search_seed, evaluation_transform=noise.transform)
    request_id = request_id_base
    root_policy_sha = None
    root_wdl_sha = None
    root_mask = core.prepare_legal_actions(state)
    step = session.advance()
    request_count = 0
    while isinstance(step, SearchEvaluationRequest):
        adapter.shared_memory.write_input((step.state, step.legal_context), shared_input[0])
        request_id += 1
        request = _Request(
            worker_id=0, lane_id=0, request_id=request_id, slot_ids=(0,),
            broker_received_at=time.perf_counter())
        broker._dispatch((request,))
        response = response_queue.get_nowait()
        if response.request_id != request_id or response.slot_ids != (0,) or response.error:
            raise RuntimeError(f"MCTS inference response identity mismatch: {response}")
        policy = shared_policy[0].detach().clone()
        wdl = shared_wdl[0].detach().clone()
        if request_count == 0:
            root_policy_sha = _tensor_sha256(policy, torch)
            root_wdl_sha = _tensor_sha256(wdl, torch)
        request_count += 1
        step = session.resume(Evaluation(
            policy=tuple(float(value) for value in policy.tolist()),
            wdl=tuple(float(value) for value in wdl.tolist())))
    if not hasattr(step, "root_visits"):
        raise RuntimeError(f"MCTS session did not return a root result: {type(step).__name__}")

    # Rewind the production per-game action RNG through previously played roots.
    # This proves that identical root visits and RNG identity select the same
    # played move at temperature 1 or temperature 0.
    rng = random.Random(int(raw["game_seed"]))
    game_positions = raw["positions"]
    target_index = next(i for i, candidate in enumerate(game_positions)
                        if int(candidate["ply"]) == int(position["ply"]))
    for earlier in game_positions[:target_index]:
        earlier_state = core.torus9_state_from_identity(earlier["state"])
        legal = core.prepare_legal_actions(earlier_state)
        replay = SearchResult(
            action=earlier["selected_action"], legal_actions=tuple(legal.actions),
            root_visits=tuple(int(value) for value in earlier["root_visits"]),
            pi=tuple(float(value) for value in earlier["pi"]),
            simulations=sum(int(value) for value in earlier["root_visits"]),
            evaluator_calls=0, legal_action_mask=tuple(legal.action_mask))
        temperature = 1.0 if int(earlier["ply"]) <= 8 else 0.0
        sampled = sample_action_from_search_result(
            replay, temperature=temperature, rng=rng, action_index=core._action_index)
        if sampled != earlier["selected_action"]:
            raise ValueError("Could not reproduce the production action RNG prefix")
    temperature = 1.0 if int(position["ply"]) <= 8 else 0.0
    selected = sample_action_from_search_result(
        step, temperature=temperature, rng=rng, action_index=core._action_index)
    return {
        "identity": example["identity"], "labels": example["labels"],
        "network_policy_sha256": root_policy_sha,
        "network_wdl_sha256": root_wdl_sha,
        "root_visits": [int(value) for value in step.root_visits],
        "pi": [float(value) for value in step.pi],
        "mcts_greedy_action": step.action,
        "selected_move": selected,
        "recorded_root_visits": [int(value) for value in position["root_visits"]],
        "recorded_pi": [float(value) for value in position["pi"]],
        "recorded_selected_action": position["selected_action"],
        "temperature": temperature,
        "simulation_cap": int(step.simulations),
        "legal_action_mask_sha256": sha256_json([bool(v) for v in root_mask.action_mask]),
        "legal_action_count": sum(bool(v) for v in root_mask.action_mask),
        "search_seed": search_seed,
        "dirichlet_seed": derive_seed(search_seed, "dirichlet"),
        "inference_requests": request_count,
        "matches_recorded_root": (
            [int(value) for value in step.root_visits] == [int(value) for value in position["root_visits"]]
            and [float(value) for value in step.pi] == [float(value) for value in position["pi"]]
            and selected == position["selected_action"]
        ),
    }


def worker_integration(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    runtime = _runtime(repo, expected_revision=args.expected_revision, device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch in integration worker")
    _, model, model_hash_value = _load_model(checkpoint, torch, "cuda")
    from gocube_golden import torus9_monolith as core
    from gocube_golden.provenance import derive_seed
    from gocube_golden.search import SearchEvaluationRequest
    import gocube_golden.search as search_module
    from gocube_golden.torus9_adaptation import AdaptationSearchContract
    import selfplay_engine as engine_module
    _CentralInference = engine_module._CentralInference
    _Request = engine_module._Request

    examples = _position_examples(Path(args.games_jsonl), core)
    adapter = _direct_adapter(model, checkpoint, FIXED_RUN_ID, MASTER_SEED)
    count = len(examples)
    slots = torch.zeros((count, 5, 81), dtype=torch.float32).share_memory_()
    expected_observations = []
    legal_by_identity = {}
    for index, example in enumerate(examples):
        state = core.torus9_state_from_identity(example["position"]["state"])
        legal = core.prepare_legal_actions(state)
        adapter.shared_memory.write_input((state, legal), slots[index])
        expected_observations.append(slots[index].clone())
        legal_by_identity[example["identity"]] = [bool(value) for value in legal.action_mask]
    policies = torch.full((count, 82), -99.0, dtype=torch.float32).share_memory_()
    wdls = torch.full((count, 3), -99.0, dtype=torch.float32).share_memory_()
    responses = [[queue.Queue()]]
    broker = _CentralInference(
        queue.Queue(), responses, None, batch_cap=64, wait_ms=1.0, device="cuda",
        shared_spec=adapter.shared_memory, infer_shared_batch=adapter.infer_shared_batch,
        shared_inputs=[slots], shared_policy=[policies], shared_wdl=[wdls])

    # Deliberately permute output rows across two requests and non-contiguous
    # shared slots. Each response must keep its request and slot identity.
    slot_order = list(range(count - 1, -1, -1))
    split = max(1, count // 2)
    grouped_slots = (tuple(slot_order[:split]), tuple(slot_order[split:]))
    request_specs = []
    for ordinal, ids in enumerate(grouped_slots):
        if ids:
            request_specs.append(_Request(0, 0, 7001 + ordinal, slot_ids=ids,
                                          broker_received_at=time.perf_counter()))
    flattened_slots = [slot for request in request_specs for slot in request.slot_ids]
    flattened = torch.stack([slots[slot] for slot in flattened_slots])
    direct = adapter.infer_shared_batch(flattened)
    direct_policy, direct_wdl = direct.policy, direct.wdl
    broker._dispatch(tuple(request_specs))
    mapping_rows = []
    direct_row = 0
    for request in request_specs:
        response = responses[0][0].get_nowait()
        if response.request_id != request.request_id or response.slot_ids != request.slot_ids or response.error:
            raise RuntimeError(f"Shared dispatch changed request identity: {response}")
        for slot in request.slot_ids:
            example = examples[slot]
            if not torch.equal(policies[slot], direct_policy[direct_row].detach().cpu()) or not torch.equal(
                    wdls[slot], direct_wdl[direct_row].detach().cpu()):
                raise RuntimeError(f"Shared output row mapped to the wrong request/slot: {example['identity']}")
            mask = legal_by_identity[example["identity"]]
            masked = _masked_policy(policies[slot], mask, torch)
            mapping_rows.append({
                "request_id": int(request.request_id), "worker_id": int(request.worker_id),
                "lane_id": int(request.lane_id), "row": direct_row, "slot_id": int(slot),
                "game_position_identity": example["identity"],
                "observation_sha256": _tensor_sha256(slots[slot], torch),
                "legal_mask_sha256": sha256_json(mask),
                "policy_sha256": _tensor_sha256(policies[slot], torch),
                "wdl_sha256": _tensor_sha256(wdls[slot], torch),
                "legal_masked_policy_sha256": _tensor_sha256(masked, torch),
            })
            direct_row += 1
    if direct_row != count:
        raise RuntimeError("Shared dispatch lost one or more real M233 request rows")

    contract = AdaptationSearchContract(simulations=FIXED_SIMULATIONS, komi=1.5)
    if not hasattr(core, "Torus9RootNoiseEvaluator"):
        raise RuntimeError("Selected Torus9 source has no root-noise production evaluator")
    # The reference output row above is produced through the actual #231 parent
    # dispatcher. Run full PUCT at each root by routing every leaf request
    # through that same dispatcher and shared-slot codec.
    search_results = []
    for example_index, example in enumerate(examples):
        state = core.torus9_state_from_identity(example["position"]["state"])
        search_results.append(_mcts_via_shared_dispatch(
            state=state, example=example, request_id_base=1000000 + example_index * 1000000,
            adapter=adapter, broker=broker,
            shared_input=slots, shared_policy=policies, shared_wdl=wdls,
            response_queue=responses[0][0], torch=torch, core=core,
            contract=contract, search_module=search_module))
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": sha256_file(checkpoint),
        "model_hash": model_hash_value, "gpu": runtime["gpu"], "cuda": runtime["cuda"],
        "representative_positions": len(examples),
        "categories": {},
        "row_request_mapping": mapping_rows,
        "mapping_rows": len(mapping_rows), "lost_or_misrouted_rows": 0,
        "mcts_roots": search_results,
        "all_roots_match_recorded_selfplay": all(
            result["matches_recorded_root"] for result in search_results),
        "policy_wdl_strict": True,
    }
    summary["categories"] = {
        label: [example["identity"] for example in examples if label in example["labels"]]
        for label in ("opening", "early", "middle", "late", "small_legal_set")
    }
    write_json(output / "integration.json", summary)
    write_json(output / "summary.json", {k: v for k, v in summary.items()
                                         if k not in ("row_request_mapping", "mcts_roots")})
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("row_request_mapping", "mcts_roots")}, indent=2), flush=True)
    return summary


class _TracingRandom:
    instances: list["_TracingRandom"] = []

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self.draws: list[int] = []
        self.instances.append(self)

    def randrange(self, *args: int) -> int:
        value = self._random.randrange(*args)
        self.draws.append(value)
        return value


def _model_state_hash(model: Any, torch: Any) -> str:
    rows = []
    for name, tensor in model.state_dict().items():
        rows.append({"name": name, "sha256": _tensor_sha256(tensor, torch)})
    return sha256_json(rows)


def _adam_state_summary(trainer: Any, torch: Any) -> dict[str, Any]:
    rows = []
    for name, parameter in trainer.model.named_parameters():
        state = trainer.optimizer.state[parameter]
        rows.append({
            "name": name,
            "step": int(state["step"]),
            "exp_avg_sha256": _tensor_sha256(state["exp_avg"], torch),
            "exp_avg_sq_sha256": _tensor_sha256(state["exp_avg_sq"], torch),
        })
    return {"parameters": rows, "sha256": sha256_json(rows)}


def worker_training(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    runtime = _runtime(repo, expected_revision=args.expected_revision, device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch in optimizer worker")
    dataset_path = Path(args.dataset).resolve()
    dataset_raw = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset_raw.get("contract") is None or not dataset_raw.get("games"):
        raise ValueError("Bounded learner dataset is empty or has no target contract")
    games = dataset_raw["games"]
    if dataset_raw.get("actor_hash") != _load_model(checkpoint, torch, "cpu")[2]:
        raise ValueError("Bounded learner dataset actor differs from M233")
    sample_count = sum(len(game["score"]) for game in games)
    if sample_count < 64:
        raise ValueError(f"32 updates require at least 64 bounded learner samples, found {sample_count}")

    from gocube_golden.orchestrator_v2.execution_permit import _test_authority
    import gocube_golden.torus9_five_channel_training as ordinary
    import types

    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ordinary.OrdinaryTrainer(
        checkpoint, learning_rate=1e-4, seed=int(raw_checkpoint["seed"]),
        gradient_clip=8.0, device="cuda")
    initial_optimizer = _adam_state_summary(trainer, torch)
    initial_model = _model_state_hash(trainer.model, torch)
    cumulative = []
    count = 0
    for game in games:
        count += len(game["score"])
        cumulative.append(count)
    sample_ids_by_game = [list(game["sample_ids"]) for game in games]
    _TracingRandom.instances = []
    original_random = ordinary.random
    ordinary.random = types.SimpleNamespace(Random=_TracingRandom)
    steps = []
    try:
        with _test_authority():
            for number in range(32):
                _TracingRandom.instances = []
                step = trainer.step(games)
                if len(_TracingRandom.instances) != 1 or len(_TracingRandom.instances[0].draws) != 64:
                    raise RuntimeError("Could not recover the production trainer's 64 sampled row identities")
                sampled_indices = _TracingRandom.instances[0].draws
                row_ids = []
                for sample_index in sampled_indices:
                    game_index = __import__("bisect").bisect_right(cumulative, sample_index)
                    previous = cumulative[game_index - 1] if game_index else 0
                    row_ids.append(sample_ids_by_game[game_index][sample_index - previous])
                step = dict(step)
                step["sampled_row_ids"] = row_ids
                step["clipped"] = float(step["grad_norm_before_clip"]) > float(trainer.gradient_clip)
                steps.append(step)
                if (number + 1) % 8 == 0:
                    print(f"[{runtime['revision'][:8]}] deterministic optimizer updates {number + 1}/32",
                          flush=True)
    finally:
        ordinary.random = original_random
    torch.cuda.synchronize()
    final_model = _model_state_hash(trainer.model, torch)
    final_optimizer = _adam_state_summary(trainer, torch)
    if trainer.update != int(raw_checkpoint.get("ordinary_update", 0)) + 32:
        raise ValueError("Bounded optimizer update counter drift")
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": sha256_file(checkpoint),
        "model_hash_initial": initial_model, "model_hash_final": final_model,
        "initial_adam_state": initial_optimizer, "final_adam_state": final_optimizer,
        "updates": trainer.update - int(raw_checkpoint.get("ordinary_update", 0)),
        "initial_update": int(raw_checkpoint.get("ordinary_update", 0)),
        "final_update": int(trainer.update), "seed": int(raw_checkpoint["seed"]),
        "device": "cuda", "gpu": runtime["gpu"], "cuda": runtime["cuda"],
        "dataset_sha256": sha256_file(dataset_path), "sample_count": sample_count,
        "steps": steps,
        "loss_components": sorted(steps[0]["losses"]),
        "all_rows_resolved": all(len(step["sampled_row_ids"]) == 64 for step in steps),
        "m233_checkpoint_unchanged": sha256_file(checkpoint) == args.expected_checkpoint_sha,
    }
    write_json(output / "training.json", summary)
    write_json(output / "summary.json", {k: v for k, v in summary.items()
                                         if k not in ("steps", "initial_adam_state", "final_adam_state")})
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("steps", "initial_adam_state", "final_adam_state")},
                     sort_keys=True, indent=2), flush=True)
    return summary


def _pcr_request(checkpoint_ref: Mapping[str, Any], games: int) -> dict[str, Any]:
    config = {
        "audit": "scientific-continuation-gate-pcr-v1",
        "topology": "torus9", "observation_channels": 5, "komi": 1.5,
        "search_mode": "pcr", "pcr": PCR_CONFIG,
        "cpuct": 1.25, "fpu": 0.0, "root_noise": True,
        "dirichlet_epsilon": 0.25, "dirichlet_alpha": 0.11,
        "temperature_until_ply": 8, "temperature_after": 0.0,
        "resign": False, "device": "cuda",
    }
    config_fingerprint = "sha256:" + sha256_json(config)
    return {"parent": dict(checkpoint_ref), "config": config_fingerprint,
            "generation": 1, "offset": 0, "games": games}


def _production_pcr_targets(records: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from gocube_golden.torus9_adaptation import FINGERPRINT, game_targets
    from gocube_golden.torus9_pcr import position_telemetry
    targets = []
    for record in records:
        target = game_targets(record)
        if target is not None:
            target["split"] = "train"
            targets.append(target)
    return targets, position_telemetry(records, __import__(
        "gocube_golden.torus9_pcr", fromlist=["*"]).PlayoutCapRandomization(**PCR_CONFIG))


def _write_pcr_resume_shard(output: Path, records: Sequence[Any], result: Any,
                            target_games: Sequence[Mapping[str, Any]],
                            pcr_telemetry: Mapping[str, Any],
                            checkpoint_ref: Mapping[str, Any]) -> dict[str, Any]:
    from gocube_golden.torus9_adaptation import FINGERPRINT, save_torch
    from gocube_golden.neural import model_hash

    shard_root = output / "production-shard"
    shard_root.mkdir(parents=True, exist_ok=False)
    generation, offset = 1, 0
    count = len(records)
    raw_path = shard_root / f"g{generation:04d}-{offset:04d}.games.jsonl.gz"
    raw_tmp = raw_path.with_suffix(".tmp")
    with gzip.open(raw_tmp, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict()) + "\n")
    os.replace(raw_tmp, raw_path)
    raw_sha = sha256_file(raw_path)
    target_path = shard_root / f"g{generation:04d}-{offset:04d}.pt"
    config_fingerprint = _pcr_request(checkpoint_ref, count)["config"]
    save_torch(target_path, {
        "contract": FINGERPRINT,
        "actor_hash": records[0].model_hash,
        "games": list(target_games),
        "raw_games_sha": raw_sha,
        "generation": generation,
        "selfplay_simulations": None,
        "search_mode": "pcr",
        "pcr": dict(PCR_CONFIG),
    })
    expected = _pcr_request(checkpoint_ref, count)
    shard = {"path": str(target_path), "sha": sha256_file(target_path), "games": count}
    telemetry = dict(result.telemetry)
    telemetry.update(pcr_telemetry)
    write_json(target_path.with_suffix(".telemetry.json"), telemetry)
    identity = {"request": expected, "shard": shard,
                "pcr_telemetry": dict(pcr_telemetry)}
    identity_path = target_path.with_suffix(".identity.json")
    write_json(identity_path, identity)
    return {
        "request": expected,
        "shard": shard,
        "raw_games_path": str(raw_path),
        "raw_games_sha256": raw_sha,
        "target_path": str(target_path),
        "target_sha256": sha256_file(target_path),
        "identity_path": str(identity_path),
        "identity_sha256": sha256_file(identity_path),
        "telemetry_path": str(target_path.with_suffix(".telemetry.json")),
        "pcr_telemetry": dict(pcr_telemetry),
    }


def _mode_decisions(wrapped_games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for wrapped in wrapped_games:
        semantic = wrapped["semantic"]
        for position in semantic["positions"]:
            decisions.append({
                "game_id": semantic["game_id"], "ply": position["ply"],
                "game_seed": semantic["game_seed"], "mode": position["search_mode"],
                "simulation_cap": position["simulation_cap"],
                "training_eligible": position["training_eligible"],
            })
    return sorted(decisions, key=lambda item: (item["game_id"], item["ply"]))


def compare_pcr_decisions_by_position(
        left_games: Mapping[str, Mapping[str, Any]],
        right_games: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare PCR contracts on positions observed under both worker schedules.

    Different concurrent schedules can alter a move and therefore the later
    trajectory. Positions after that trajectory split have no counterpart;
    report them explicitly while checking every shared (game, ply) identity.
    """
    def index(games: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        positions: dict[tuple[str, int], dict[str, Any]] = {}
        for game_id, wrapped in games.items():
            semantic = wrapped["semantic"]
            for position in semantic["positions"]:
                key = (str(game_id), int(position["ply"]))
                if key in positions:
                    raise ValueError(f"Duplicate PCR mode decision at {key}")
                positions[key] = {
                    "game_seed": int(semantic["game_seed"]),
                    "search_mode": str(position["search_mode"]),
                    "simulation_cap": int(position["simulation_cap"]),
                    "training_eligible": bool(position["training_eligible"]),
                }
        return positions

    left, right = index(left_games), index(right_games)
    shared = sorted(set(left) & set(right))
    differences = [
        {"game_id": key[0], "ply": key[1], "left": left[key], "right": right[key]}
        for key in shared if left[key] != right[key]
    ]
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    return {
        "identical_on_shared_positions": not differences,
        "same_game_id_set": set(left_games) == set(right_games),
        "left_games": len(left_games),
        "right_games": len(right_games),
        "shared_positions": len(shared),
        "left_only_positions": len(left_only),
        "right_only_positions": len(right_only),
        "left_only_examples": [{"game_id": game_id, "ply": ply}
                               for game_id, ply in left_only[:5]],
        "right_only_examples": [{"game_id": game_id, "ply": ply}
                                for game_id, ply in right_only[:5]],
        "mismatch_count": len(differences),
        "first_mismatch": differences[0] if differences else None,
        "left_decision_sha256": sha256_json([
            {"game_id": game_id, "ply": ply, **left[(game_id, ply)]}
            for game_id, ply in sorted(left)]),
        "right_decision_sha256": sha256_json([
            {"game_id": game_id, "ply": ply, **right[(game_id, ply)]}
            for game_id, ply in sorted(right)]),
    }


def _pcr_run_invariants(games: Mapping[str, Mapping[str, Any]],
                        target_summary: Mapping[str, Any],
                        worker_summary: Mapping[str, Any]) -> dict[str, Any]:
    from gocube_golden.torus9_pcr import PlayoutCapRandomization

    caps = PlayoutCapRandomization(**PCR_CONFIG)
    expected_full = {
        f"{row['game_id']}:{int(position['ply']):04d}"
        for row in games.values()
        for position in row["raw_record"]["positions"]
        if position.get("search_mode") == "full"
    }
    cheap_ids = {
        f"{row['game_id']}:{int(position['ply']):04d}"
        for row in games.values()
        for position in row["raw_record"]["positions"]
        if position.get("search_mode") == "cheap"
    }
    actual = {str(row["row_id"]) for row in target_summary["samples"]}
    cheap_leakage = len(actual & cheap_ids)
    missing_full = len(expected_full - actual)
    unexpected = len(actual - expected_full)
    technical = sum(row["raw_record"].get("technical_termination") is not None
                    for row in games.values())
    roots = worker_summary["root_contract"]
    positions = [(row, position) for row in games.values()
                 for position in row["raw_record"]["positions"]]
    contract_ok = (
        roots["invalid_cap_noise_eligibility_combinations"] == 0
        and roots["cheap_roots"] + roots["full_roots"] == roots["roots"]
        and all(position.get("search_mode") in {"cheap", "full"}
                and int(position.get("search_simulations") or 0) ==
                    (PCR_CONFIG["cheap_simulations"] if position["search_mode"] == "cheap"
                     else PCR_CONFIG["full_simulations"])
                and bool(position.get("training_eligible")) ==
                    (position["search_mode"] == "full")
                for row, position in positions)
    )
    selection_ok = all(
        position.get("search_mode") == caps.mode(
            int(row["raw_record"]["game_seed"]), int(position["ply"]))
        for row, position in positions)
    target_ok = actual == expected_full and cheap_leakage == 0
    return {
        "passed": (contract_ok and selection_ok and target_ok and technical == 0
                   and target_summary["sample_count"] == len(expected_full)),
        "root_contract_valid": contract_ok,
        "selection_matches_game_ply_seed": selection_ok,
        "roots": roots["roots"], "cheap_roots": roots["cheap_roots"],
        "full_roots": roots["full_roots"],
        "invalid_cap_noise_eligibility_combinations":
            roots["invalid_cap_noise_eligibility_combinations"],
        "learner_samples": target_summary["sample_count"],
        "expected_full_learner_positions": len(expected_full),
        "cheap_learner_leakage": cheap_leakage,
        "missing_full_learner_positions": missing_full,
        "unexpected_learner_positions": unexpected,
        "technical_games": technical,
        "all_cheap_games_observed": sum(
            not any(position.get("search_mode") == "full"
                    for position in row["raw_record"]["positions"])
            for row in games.values()),
    }


def _validate_played_move_schedule(wrapped_games: Sequence[Mapping[str, Any]], core: Any) -> int:
    from gocube_golden.search import SearchResult
    from gocube_golden.selfplay_policy import sample_action_from_search_result
    checked = 0
    for wrapped in wrapped_games:
        raw = wrapped["raw_record"]
        rng = random.Random(int(raw["game_seed"]))
        for position in raw["positions"]:
            state = core.torus9_state_from_identity(position["state"])
            legal = core.prepare_legal_actions(state)
            search_result = SearchResult(
                action=position["selected_action"], legal_actions=tuple(legal.actions),
                root_visits=tuple(int(value) for value in position["root_visits"]),
                pi=tuple(float(value) for value in position["pi"]),
                simulations=sum(int(value) for value in position["root_visits"]),
                evaluator_calls=0, legal_action_mask=tuple(legal.action_mask))
            temperature = 1.0 if int(position["ply"]) <= 8 else 0.0
            selected = sample_action_from_search_result(
                search_result, temperature=temperature, rng=rng,
                action_index=core._action_index)
            if selected != position["selected_action"]:
                raise ValueError(f"Played-move RNG/temperature drift at {raw['game_id']}:{position['ply']}")
            if selected == "resign":
                raise ValueError("Torus9 self-play unexpectedly resigned")
            checked += 1
    return checked


def _read_production_raw(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: malformed raw game record")
                rows.append(value)
    return rows


def worker_resume_check(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    runtime = _runtime(repo, expected_revision=args.expected_revision, device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch while resuming test shard")
    from gocube_golden import torus9_monolith as core
    shard_root = Path(args.shard_root).resolve()
    identity_path = shard_root / "g0001-0000.identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    checkpoint_ref = json.loads(Path(args.checkpoint_ref).read_text(encoding="utf-8"))["checkpoint"]
    expected = _pcr_request(checkpoint_ref, args.games)
    if identity.get("request") != expected:
        raise ValueError("PCR test shard resume request identity drift")
    shard = identity["shard"]
    target_path = shard_root / "g0001-0000.pt"
    raw_path = shard_root / "g0001-0000.games.jsonl.gz"
    before = {str(path): sha256_file(path) for path in (identity_path, target_path, raw_path)}
    if Path(shard["path"]).resolve() != target_path.resolve() or sha256_file(target_path) != shard["sha"]:
        raise ValueError("PCR test shard target artifact SHA drift")
    target_raw = torch.load(target_path, map_location="cpu", weights_only=False)
    if target_raw.get("search_mode") != "pcr" or target_raw.get("pcr") != PCR_CONFIG:
        raise ValueError("Resumed test shard PCR contract drift")
    if sha256_file(raw_path) != target_raw.get("raw_games_sha"):
        raise ValueError("Resumed raw game shard SHA drift")
    raw_records = _read_production_raw(raw_path)
    wrapped = []
    for raw in raw_records:
        semantic = semantic_game_from_record(raw, core)
        wrapped.append({"game_id": raw["game_id"], "semantic": semantic,
                        "semantic_sha256": sha256_json(semantic), "raw_record": raw})
    targets = target_raw["games"]
    target_summary = _target_summaries(wrapped, targets, core, torch)
    decisions = _mode_decisions(wrapped)
    roots = []
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": sha256_file(checkpoint),
        "games": len(wrapped), "plies": sum(row["semantic"]["plies"] for row in wrapped),
        "semantic_games": [{"game_id": row["game_id"], "sha256": row["semantic_sha256"]}
                           for row in wrapped],
        "semantic_games_sha256": sha256_json([
            {"game_id": row["game_id"], "sha256": row["semantic_sha256"]} for row in wrapped]),
        "learner_samples": target_summary["sample_count"],
        "targets": target_summary,
        "mode_decisions": decisions,
        "mode_decisions_sha256": sha256_json(decisions),
        "request": identity["request"],
        "shard_identity": identity,
        "pcr_telemetry": identity["pcr_telemetry"],
        "committed_shard_loaded_read_only": True,
        "input_shard_sha256_before_after": before,
        "input_shard_unchanged": before == {str(path): sha256_file(path)
                                            for path in (identity_path, target_path, raw_path)},
        "all_pcr_positions_represented": len(decisions) == sum(
            len(row["raw_record"]["positions"]) for row in wrapped),
    }
    write_json(output / "resume.json", summary)
    write_json(output / "summary.json", {k: v for k, v in summary.items()
                                         if k not in ("targets", "mode_decisions", "shard_identity")})
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("targets", "mode_decisions", "shard_identity")},
                     sort_keys=True, indent=2), flush=True)
    return summary


def worker_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(Path(args.repo).resolve(), expected_revision=args.expected_revision,
                       device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch in divergence diagnostic")
    _, model, model_hash_value = _load_model(checkpoint, torch, "cuda")
    from gocube_golden import torus9_monolith as core
    from gocube_golden.torus9_adaptation import AdaptationSelfPlayAdapter
    state_identity = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    state = core.torus9_state_from_identity(state_identity)
    legal = core.prepare_legal_actions(state)
    adapter = AdaptationSelfPlayAdapter(
        model, checkpoint=checkpoint, run_id=FIXED_RUN_ID, seed=MASTER_SEED,
        device="cuda", simulations=FIXED_SIMULATIONS)
    observation = torch.empty((1, 5, 81), dtype=torch.float32)
    adapter.shared_memory.write_input((state, legal), observation[0])
    result = adapter.infer_shared_batch(observation)
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": sha256_file(checkpoint),
        "model_hash": model_hash_value, "state_identity": state_identity,
        "state_sha256": sha256_json(state_identity),
        "legal_mask": [bool(value) for value in legal.action_mask],
        "legal_actions": [_json_action(action) for action in legal.actions],
        "legal_mask_sha256": sha256_json([bool(value) for value in legal.action_mask]),
        "network_policy": [float(value) for value in result.policy[0].tolist()],
        "network_wdl": [float(value) for value in result.wdl[0].tolist()],
        "network_policy_sha256": _tensor_sha256(result.policy[0], torch),
        "network_wdl_sha256": _tensor_sha256(result.wdl[0], torch),
        "rng_identity": {"master_seed": args.master_seed, "game_seed": args.game_seed,
                         "ply": args.ply, "search_seed": args.search_seed,
                         "dirichlet_seed": __import__("gocube_golden.provenance", fromlist=["derive_seed"])
                         .derive_seed(args.search_seed, "dirichlet")},
        "raw_left_root": json.loads(Path(args.left_root_json).read_text(encoding="utf-8")),
        "raw_right_root": json.loads(Path(args.right_root_json).read_text(encoding="utf-8")),
    }
    write_json(output / "first-divergence.json", summary)
    return summary


def _serialized_request_identity(request: Any, *, index: int, core: Any,
                                torch: Any) -> dict[str, Any]:
    observation_sha, _blob = _trace_observation(request, torch)
    identity = core.torus9_state_identity(request.state)
    legal_mask = [bool(value) for value in request.legal_context.action_mask]
    return {"session_request_index": int(index),
            "state_sha256": sha256_json(identity),
            "legal_mask_sha256": sha256_json(legal_mask),
            "observation_sha256": observation_sha}


def worker_replay_evaluations(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(Path(args.repo).resolve(), expected_revision=args.expected_revision,
                       device="cuda")
    torch = runtime["torch"]
    from gocube_golden import torus9_monolith as core
    from gocube_golden.arena_contract import SearchSettings
    from gocube_golden import search as search_module
    from gocube_golden.search import Evaluation, SearchEvaluationRequest, SearchResult
    stream = json.loads(Path(args.stream_json).read_text(encoding="utf-8"))
    if stream.get("schema") != _DIAGNOSTICS.EVALUATION_STREAM_SCHEMA:
        raise ValueError("unsupported fixed Evaluation stream schema")
    state = core.torus9_state_from_identity(stream["root_identity"]["identity"])
    settings_data = stream["settings"]
    settings = SearchSettings(
        simulations=int(stream["simulation_cap"]),
        cpuct=float(settings_data["cpuct"]), fpu=float(settings_data["fpu"]),
        deterministic_tie_break=bool(settings_data["deterministic_tie_break"]))
    noise_data = stream["noise"]
    noise = core.Torus9RootNoiseEvaluator(
        None, state, seed=int(noise_data["seed"]), alpha=float(noise_data["alpha"]))
    session = search_module.SequentialPUCTSession(
        state, settings, adapter=core.GoldenSearchAdapter(),
        seed=int(stream["search_seed"]), evaluation_transform=noise.transform)
    step: Any = session.advance()
    observed_requests = []
    request_sequence_matches = True
    divergence = None
    for expected in stream["evaluations"]:
        if not isinstance(step, SearchEvaluationRequest):
            request_sequence_matches = False
            divergence = {"request_index": len(observed_requests),
                          "expected": expected.get("request"),
                          "actual": "search completed before stream was exhausted"}
            break
        actual = _serialized_request_identity(
            step, index=len(observed_requests), core=core, torch=torch)
        observed_requests.append(actual)
        if actual != expected["request"] and divergence is None:
            request_sequence_matches = False
            divergence = {"request_index": len(observed_requests) - 1,
                          "expected": expected["request"], "actual": actual}
        evaluation = Evaluation(
            policy=tuple(float(value) for value in expected["policy"]),
            wdl=tuple(float(value) for value in expected["wdl"]))
        step = session.resume(evaluation)
    completed = isinstance(step, SearchResult)
    if not completed:
        request_sequence_matches = False
        divergence = divergence or {
            "request_index": len(observed_requests),
            "expected": "end of stream", "actual": type(step).__name__}
    result = None
    if completed:
        result = {"root_visits": [int(value) for value in step.root_visits],
                  "pi": [float(value) for value in step.pi],
                  "selected_action": _json_action(step.action),
                  "simulations": int(step.simulations),
                  "evaluator_calls": int(step.evaluator_calls)}
    summary = {
        "source_revision": runtime["revision"],
        "stream_sha256": sha256_file(Path(args.stream_json)),
        "stream_evaluations": len(stream["evaluations"]),
        "observed_requests": len(observed_requests),
        "request_sequence_matches": request_sequence_matches,
        "first_request_difference": divergence,
        "completed": completed, "result": result,
        "result_sha256": None if result is None else sha256_json(result),
    }
    write_json(output / "replay.json", summary)
    write_json(output / "requested-leaves.json", observed_requests)
    return summary


def _observation_from_identity(identity: Mapping[str, Any], core: Any,
                               torch: Any) -> Any:
    from gocube_golden.torus9_adaptation import write_observation
    state = core.torus9_state_from_identity(identity)
    legal = core.prepare_legal_actions(state)
    observation = torch.empty((5, 81), dtype=torch.float32)
    write_observation((state, legal), observation)
    return observation


def worker_batch_geometry(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(Path(args.repo).resolve(), expected_revision=args.expected_revision,
                       device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch in batch geometry worker")
    _raw, model, model_hash_value = _load_model(checkpoint, torch, "cuda")
    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    request_index = int(args.request_index)
    requests = trace["evaluation_requests"]
    if request_index < 0 or request_index >= len(requests):
        raise ValueError("frozen observation request index is outside the root trace")
    selected = requests[request_index]
    target_bytes = bytes.fromhex(selected["observation_blob_hex"])
    target = torch.frombuffer(bytearray(target_bytes), dtype=torch.float32).clone().reshape(5, 81)
    if _tensor_sha256(target, torch) != selected["observation_sha256"]:
        raise ValueError("serialized frozen observation SHA does not match its trace")

    from gocube_golden import torus9_monolith as core
    filler_observations = []
    seen = set()
    for row in sorted(read_jsonl(Path(args.filler_games_jsonl)),
                      key=lambda item: str(item["game_id"])):
        raw = row["raw_record"]
        for position in raw["positions"]:
            identity = position["state"]
            identity_sha = sha256_json(identity)
            if identity_sha in seen or identity_sha == trace["root_identity"]["state_sha256"]:
                continue
            seen.add(identity_sha)
            filler_observations.append(_observation_from_identity(identity, core, torch))
            if len(filler_observations) >= 96:
                break
        if len(filler_observations) >= 96:
            break
    if not filler_observations:
        raise ValueError("batch geometry requires real Torus9 filler observations")

    actual_geometry = selected.get("inference_batch")

    def variant_rows(batch_size: int) -> list[int]:
        if batch_size == 1:
            return [0]
        rows = [0, batch_size // 2, batch_size - 1]
        if actual_geometry and int(actual_geometry["batch_size"]) == batch_size:
            rows.append(int(actual_geometry["row_index"]))
        return list(dict.fromkeys(rows))

    def forward(batch_cpu: Any, repeat: int, batch_size: int, target_row: int) -> dict[str, Any]:
        batch = batch_cpu.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        with torch.inference_mode():
            raw_policy, raw_wdl = model(batch)
            policy = torch.softmax(raw_policy, dim=1)
            wdl = torch.softmax(raw_wdl, dim=1)
        torch.cuda.synchronize()
        policy_logits = raw_policy[target_row].detach().cpu().contiguous()
        wdl_logits = raw_wdl[target_row].detach().cpu().contiguous()
        policy_row = policy[target_row].detach().cpu().contiguous()
        wdl_row = wdl[target_row].detach().cpu().contiguous()
        return {
            "scope": args.scope, "repeat": int(repeat), "pid": os.getpid(),
            "batch_size": int(batch_size), "target_row": int(target_row),
            "target_observation_sha256": selected["observation_sha256"],
            "batch_observation_sha256": [
                _tensor_sha256(batch_cpu[row], torch) for row in range(batch_size)],
            "policy_logits": [float(value) for value in policy_logits.tolist()],
            "wdl_logits": [float(value) for value in wdl_logits.tolist()],
            "policy": [float(value) for value in policy_row.tolist()],
            "wdl": [float(value) for value in wdl_row.tolist()],
            "policy_logits_sha256": _tensor_sha256(policy_logits, torch),
            "wdl_logits_sha256": _tensor_sha256(wdl_logits, torch),
            "policy_sha256": _tensor_sha256(policy_row, torch),
            "wdl_sha256": _tensor_sha256(wdl_row, torch),
        }

    rows = []
    repeats = int(args.repeats)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if actual_geometry and int(actual_geometry["batch_size"]) not in batch_sizes:
        batch_sizes.append(int(actual_geometry["batch_size"]))
    # Warm the exact inference path; warmup is excluded from repeat evidence.
    with torch.inference_mode():
        model(target.unsqueeze(0).to("cuda"))
    torch.cuda.synchronize()
    for repeat in range(repeats):
        repeat_index = int(args.repeat_offset) + repeat
        for batch_size in batch_sizes:
            for target_row in variant_rows(batch_size):
                batch_rows = []
                for row_index in range(batch_size):
                    filler = filler_observations[(repeat_index * 67 + batch_size * 3 + row_index)
                                                 % len(filler_observations)]
                    batch_rows.append(filler.clone())
                batch_rows[target_row] = target.clone()
                rows.append(forward(torch.stack(batch_rows), repeat_index,
                                    batch_size, target_row))
    write_jsonl(output / "rows.jsonl", rows)
    summary = {
        "source_revision": runtime["revision"], "checkpoint_sha256": sha256_file(checkpoint),
        "model_hash": model_hash_value, "gpu": runtime["gpu"], "cuda": runtime["cuda"],
        "torch": torch.__version__, "scope": args.scope, "repeat_count": repeats,
        "frozen_observation_sha256": selected["observation_sha256"],
        "frozen_observation_request_index": request_index,
        "actual_inference_batch": actual_geometry,
        "filler_observation_count": len(filler_observations),
        "rows_path": str(output / "rows.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary


def worker_controlled_geometry_search(args: argparse.Namespace) -> dict[str, Any]:
    """Run real sequential PUCT with root Evaluation fixed and leaf batch shape varied."""
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = _runtime(Path(args.repo).resolve(), expected_revision=args.expected_revision,
                       device="cuda")
    torch = runtime["torch"]
    checkpoint = Path(args.checkpoint).resolve()
    if sha256_file(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("M233 checkpoint SHA mismatch in controlled geometry search")
    _raw, model, model_hash_value = _load_model(checkpoint, torch, "cuda")
    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    if trace.get("schema") != _DIAGNOSTICS.ROOT_TRACE_SCHEMA:
        raise ValueError("unsupported root trace schema")

    from gocube_golden import torus9_monolith as core
    from gocube_golden.arena_contract import SearchSettings
    from gocube_golden import search as search_module
    from gocube_golden.search import Evaluation, SearchEvaluationRequest, SearchResult

    root_identity = trace["root_identity"]
    root_state = core.torus9_state_from_identity(root_identity["identity"])
    settings_data = trace["settings"]
    settings = SearchSettings(
        simulations=int(trace["simulation_cap"]), cpuct=float(settings_data["cpuct"]),
        fpu=float(settings_data["fpu"]),
        deterministic_tie_break=bool(settings_data["deterministic_tie_break"]))
    noise_data = trace["noise"]
    tracing_session = _install_tracing_session(search_module, core, torch)

    filler_observations = []
    seen = {str(root_identity["state_sha256"])}
    for row in sorted(read_jsonl(Path(args.filler_games_jsonl)),
                      key=lambda item: str(item["game_id"])):
        for position in row["raw_record"]["positions"]:
            identity = position["state"]
            identity_sha = sha256_json(identity)
            if identity_sha in seen:
                continue
            seen.add(identity_sha)
            filler_observations.append(_observation_from_identity(identity, core, torch))
            if len(filler_observations) >= 96:
                break
        if len(filler_observations) >= 96:
            break
    if len(filler_observations) < 27:
        raise ValueError("controlled geometry search requires 27 real Torus9 filler observations")

    variants = {
        "batch1_row0": {"batch_size": 1, "target_row": 0},
        "batch28_row16": {"batch_size": 28, "target_row": 16},
        "batch16_row0": {"batch_size": 16, "target_row": 0},
    }
    traces: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    repeat_count = int(args.repeats)
    if repeat_count < 2:
        raise ValueError("controlled geometry search requires at least two repeats per variant")

    def run_one(name: str, repeat: int) -> dict[str, Any]:
        variant = variants[name]
        noise = core.Torus9RootNoiseEvaluator(
            None, root_state, seed=int(noise_data["seed"]), alpha=float(noise_data["alpha"]))
        session = tracing_session(
            root_state, settings, adapter=core.GoldenSearchAdapter(),
            seed=int(trace["search_seed"]), evaluation_transform=noise.transform)
        audit_trace = {
            "schema": _DIAGNOSTICS.ROOT_TRACE_SCHEMA,
            "game_id": trace["game_id"], "ply": int(trace["ply"]),
            "root_identity": root_identity, "search_seed": int(trace["search_seed"]),
            "simulation_cap": int(trace["simulation_cap"]),
            "settings": dict(settings_data), "noise": dict(noise_data),
            "evaluation_requests": [], "simulations": [],
        }
        session._audit_trace = audit_trace
        step = session.advance()
        request_count = 0
        while isinstance(step, SearchEvaluationRequest):
            if request_count == 0:
                batch_size, target_row = 1, 0
            else:
                batch_size, target_row = int(variant["batch_size"]), int(variant["target_row"])
            observation_sha, observation_hex = _trace_observation(step, torch)
            target_observation = torch.frombuffer(
                bytearray.fromhex(observation_hex), dtype=torch.float32).clone().reshape(5, 81)
            batch_rows = [filler_observations[index].clone() for index in range(batch_size)]
            batch_rows[target_row] = target_observation
            batch_cpu = torch.stack(batch_rows)
            torch.cuda.synchronize()
            with torch.inference_mode():
                raw_policy, raw_wdl = model(batch_cpu.to("cuda", non_blocking=False))
                policy_batch = torch.softmax(raw_policy, dim=1)
                wdl_batch = torch.softmax(raw_wdl, dim=1)
            torch.cuda.synchronize()
            policy = policy_batch[target_row].detach().cpu().contiguous()
            wdl = wdl_batch[target_row].detach().cpu().contiguous()
            step = session.resume(Evaluation(
                policy=tuple(float(value) for value in policy.tolist()),
                wdl=tuple(float(value) for value in wdl.tolist())))
            request = audit_trace["evaluation_requests"][-1]
            request["controlled_batch"] = {
                "batch_size": batch_size, "row_index": target_row,
                "target_observation_sha256": observation_sha,
                "batch_observation_sha256": [_tensor_sha256(row, torch) for row in batch_cpu],
            }
            request_count += 1
        if not isinstance(step, SearchResult):
            raise RuntimeError(f"controlled geometry search ended as {type(step).__name__}")
        audit_trace.update({
            "root_visits": [int(value) for value in step.root_visits],
            "pi": [float(value) for value in step.pi],
            "selected_action": _json_action(step.action),
            "simulations_completed": int(step.simulations),
            "evaluator_calls": int(step.evaluator_calls),
        })
        write_json(output / f"{name}-repeat-{repeat:02d}.json", audit_trace)
        return audit_trace

    for repeat in range(repeat_count):
        order = tuple(variants) if repeat % 2 == 0 else tuple(reversed(variants))
        for name in order:
            traces[name].append(run_one(name, repeat))

    pairs = (("batch28_row16", "batch16_row0"),
             ("batch1_row0", "batch28_row16"),
             ("batch1_row0", "batch16_row0"))
    pairwise = {}
    for left_name, right_name in pairs:
        pairwise[f"{left_name}_vs_{right_name}"] = (
            _DIAGNOSTICS.compare_controlled_geometry_traces(
                traces[left_name][0], traces[right_name][0],
                left_repeats=traces[left_name][1:],
                right_repeats=traces[right_name][1:]))
    comparison = pairwise["batch28_row16_vs_batch16_row0"]
    status = ("PROVEN_FOR_THIS_ROOT"
              if any(row["status"] == "PROVEN_FOR_THIS_ROOT"
                     for row in pairwise.values()) else "UNKNOWN")
    summary = {
        "schema": "torus9-controlled-root-geometry-causality-v1",
        "source_revision": runtime["revision"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_hash": model_hash_value,
        "gpu": runtime["gpu"], "cuda": runtime["cuda"], "torch": torch.__version__,
        "game_id": trace["game_id"], "ply": int(trace["ply"]),
        "root_identity": root_identity, "search_seed": int(trace["search_seed"]),
        "simulation_cap": int(trace["simulation_cap"]),
        "root_geometry": {"batch_size": 1, "row_index": 0},
        "variants": {name: {"batch_size": int(config["batch_size"]),
                            "target_row": int(config["target_row"]),
                            "repeats": len(traces[name]),
                            "trace_files": [f"{name}-repeat-{i:02d}.json"
                                            for i in range(len(traces[name]))],
                            "root_visits": row["root_visits"],
                            "selected_action": row["selected_action"],
                            "root_priors_sha256": row["root_priors_sha256"]}
                     for name, config in variants.items()
                     for row in [traces[name][0]]},
        "causality": comparison,
        "pairwise_causality": pairwise,
        "status": status,
    }
    write_json(output / "controlled-geometry-causality.json", summary)
    return summary


def run_batch_geometry(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    same = output / "same-process"
    _launch(args.python, script, "worker-batch-geometry", [
        "--repo", args.repo, "--expected-revision", args.expected_revision,
        "--checkpoint", args.checkpoint, "--expected-checkpoint-sha",
        args.expected_checkpoint_sha, "--trace-json", args.trace_json,
        "--request-index", args.request_index, "--filler-games-jsonl",
        args.filler_games_jsonl, "--output", same, "--scope", "same-process",
        "--repeats", args.same_process_repeats,
    ])
    row_paths = [same / "rows.jsonl"]
    fresh_outputs = []
    for repeat in range(args.fresh_process_repeats):
        fresh = output / f"fresh-process-{repeat:02d}"
        _launch(args.python, script, "worker-batch-geometry", [
            "--repo", args.repo, "--expected-revision", args.expected_revision,
            "--checkpoint", args.checkpoint, "--expected-checkpoint-sha",
            args.expected_checkpoint_sha, "--trace-json", args.trace_json,
            "--request-index", args.request_index, "--filler-games-jsonl",
            args.filler_games_jsonl, "--output", fresh, "--scope", "fresh-process",
            "--repeats", 1,
            "--repeat-offset", repeat,
        ])
        row_paths.append(fresh / "rows.jsonl")
        fresh_outputs.append(str(fresh))
    rows = [row for path in row_paths for row in read_jsonl(path)]
    aggregated = _DIAGNOSTICS.aggregate_batch_geometry(rows)
    summary = {
        "schema": "torus9-batch-geometry-audit-v1",
        "source_revision": args.expected_revision,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "frozen_observation_sha256": rows[0]["target_observation_sha256"],
        "same_process_repeats": args.same_process_repeats,
        "fresh_process_repeats": args.fresh_process_repeats,
        "same_process_output": str(same), "fresh_process_outputs": fresh_outputs,
        "shape_or_row_dependent": aggregated["shape_or_row_dependent"],
        "variants": aggregated["variants"],
    }
    write_jsonl(output / "all-rows.jsonl", rows)
    write_json(output / "batch-geometry.json", summary)
    return summary


def compare_root_trace_files(left_path: Path, right_path: Path,
                             output_path: Path) -> dict[str, Any]:
    output_path = ensure_tmp_path(output_path)
    left = json.loads(Path(left_path).read_text(encoding="utf-8"))
    right = json.loads(Path(right_path).read_text(encoding="utf-8"))
    report = _DIAGNOSTICS.compare_root_traces(left, right)
    report.update({"left_trace": str(left_path), "right_trace": str(right_path),
                   "left_trace_sha256": sha256_file(Path(left_path)),
                   "right_trace_sha256": sha256_file(Path(right_path))})
    write_json(output_path, report)
    return report


def run_replay_evaluations(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    trace_path = Path(args.trace_json)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    stream = _DIAGNOSTICS.serialize_evaluation_stream(trace)
    stream_path = output / "fixed-evaluation-stream.json"
    write_json(stream_path, stream)
    script = Path(__file__).resolve()
    revisions = {"A": (args.repo_a, EXPECTED_REVISIONS["A"]),
                 "B": (args.repo_b, EXPECTED_REVISIONS["B"]),
                 "C": (args.repo_c, EXPECTED_REVISIONS["C"])}
    run_dirs: dict[str, Path] = {}
    summaries: dict[str, dict[str, Any]] = {}
    request_rows: dict[str, Any] = {}
    for name in ("A1", "A2", "B", "C"):
        revision_name = "A" if name.startswith("A") else name
        repo, revision = revisions[revision_name]
        run_output = output / name
        _launch(args.python, script, "worker-replay-evaluations", [
            "--repo", repo, "--expected-revision", revision,
            "--stream-json", stream_path, "--output", run_output,
        ])
        run_dirs[name] = run_output
        summaries[name] = json.loads((run_output / "replay.json").read_text(encoding="utf-8"))
        request_rows[name] = json.loads(
            (run_output / "requested-leaves.json").read_text(encoding="utf-8"))
    expected_leaves = request_rows["A1"]
    exact = {
        name: summaries[name]["request_sequence_matches"]
              and request_rows[name] == expected_leaves
              and summaries[name]["result"] == summaries["A1"]["result"]
        for name in ("A2", "B", "C")
    }
    report = {
        "schema": "torus9-fixed-evaluation-replay-report-v1",
        "stream_sha256": sha256_file(stream_path),
        "stream_source_trace": str(trace_path),
        "evaluation_count": len(stream["evaluations"]),
        "runs": summaries,
        "exact_against_A1": exact,
        "same_revision_AA_exact": bool(exact["A2"]),
        "cross_revision_ABC_exact": bool(exact["B"] and exact["C"]),
        "status": "PASS" if all(exact.values()) else "FAIL",
    }
    write_json(output / "fixed-evaluation-replay.json", report)
    return report


def run_deterministic_core(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if int(args.games) < 16:
        raise ValueError("deterministic-core acceptance requires at least 16 complete games")
    repos = {"A": (args.repo_a, EXPECTED_REVISIONS["A"]),
             "B": (args.repo_b, EXPECTED_REVISIONS["B"]),
             "C": (args.repo_c, EXPECTED_REVISIONS["C"])}
    runs: dict[str, Path] = {}
    summaries: dict[str, dict[str, Any]] = {}
    script = Path(__file__).resolve()
    for label in ("A1", "A2", "B", "C"):
        revision_name = "A" if label.startswith("A") else label
        repo, revision = repos[revision_name]
        run_output = output / label
        _launch_selfplay(
            args.python, script, mode="fixed", repo=repo,
            expected_revision=revision, checkpoint=args.checkpoint, output=run_output,
            expected_checkpoint_sha=args.expected_checkpoint_sha,
            games=args.games, run_id=FIXED_RUN_ID, workers=1,
            active_games_per_worker=1)
        runs[label] = run_output
        summaries[label] = json.loads((run_output / "summary.json").read_text(encoding="utf-8"))
    labels = ("A1", "A2", "B", "C")
    comparisons = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            left, right = runs[left_label], runs[right_label]
            left_rows, right_rows = _load_game_rows(left / "games.jsonl"), _load_game_rows(
                right / "games.jsonl")
            comparison = _DIAGNOSTICS.compare_game_runs(
                left_rows, right_rows, f"{left_label} vs {right_label}")
            target_equal = (
                summaries[left_label]["target_field_sha256"]
                == summaries[right_label]["target_field_sha256"]
                and summaries[left_label]["learner_samples"]
                == summaries[right_label]["learner_samples"])
            exact_games = bool(comparison["same_game_id_set"]
                               and comparison["root_visit_mismatch_count"] == 0
                               and comparison["selected_action_mismatch_count"] == 0
                               and comparison["action_trace_mismatch_games"] == 0
                               and comparison["result_mismatch_games"] == 0
                               and comparison["length_mismatch_games"] == 0)
            comparisons.append({**comparison, "target_builder_exact": target_equal,
                                "complete_selfplay_exact": exact_games})
    batch_one = {label: summaries[label]["inference_batches"]["all_batches_size_one"]
                 for label in labels}
    passed = (all(item["complete_selfplay_exact"] and item["target_builder_exact"]
                  for item in comparisons) and all(batch_one.values())
              and all(summaries[label]["games"] == args.games for label in labels))
    report = {
        "schema": "torus9-deterministic-core-report-v1", "games_per_run": int(args.games),
        "run_id": FIXED_RUN_ID, "master_seed": MASTER_SEED,
        "execution": {label: summaries[label]["execution"] for label in labels},
        "inference_batches_size_one": batch_one,
        "runs": {label: {"source_revision": summaries[label]["source_revision"],
                          "games": summaries[label]["games"],
                          "roots": summaries[label]["root_contract"]["roots"],
                          "learner_samples": summaries[label]["learner_samples"],
                          "semantic_games_sha256": summaries[label]["semantic_games_sha256"],
                          "target_field_sha256": summaries[label]["target_field_sha256"],
                          "path": str(runs[label])} for label in labels},
        "comparisons": comparisons,
        "status": "PASS" if passed else "FAIL",
    }
    write_json(output / "deterministic-core.json", report)
    return report


def compare_existing_deterministic_core(args: argparse.Namespace) -> dict[str, Any]:
    """Compare completed A1/A2/B/C artifacts without rerunning any revision."""
    output = ensure_tmp_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    run_paths = {"A1": Path(args.a1), "A2": Path(args.a2),
                 "B": Path(args.b), "C": Path(args.c)}
    expected_revision = {"A1": EXPECTED_REVISIONS["A"],
                         "A2": EXPECTED_REVISIONS["A"],
                         "B": EXPECTED_REVISIONS["B"],
                         "C": EXPECTED_REVISIONS["C"]}
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    runs_report: dict[str, dict[str, Any]] = {}
    for label, run_path in run_paths.items():
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("source_revision") != expected_revision[label]:
            raise ValueError(f"{label} source revision mismatch: {summary.get('source_revision')}")
        if summary.get("checkpoint_sha256") != EXPECTED_M233_SHA256:
            raise ValueError(f"{label} M233 checkpoint SHA mismatch")
        if summary.get("games") != int(args.games):
            raise ValueError(f"{label} is incomplete: {summary.get('games')} games")
        if summary.get("technical_games") != 0:
            raise ValueError(f"{label} contains technical games")
        if not summary.get("inference_batches", {}).get("all_batches_size_one"):
            raise ValueError(f"{label} did not use batch-size-one inference throughout")
        if summary.get("execution") != {
                "active_games_per_worker": 1, "inference_batch_cap": 64,
                "inference_batch_wait_ms": 1.0, "start_method": "spawn", "workers": 1}:
            raise ValueError(f"{label} execution settings drifted")
        rows = _load_game_rows(run_path / "games.jsonl")
        if len(rows) != int(args.games):
            raise ValueError(f"{label} games.jsonl is incomplete")
        if any(row["raw_record"].get("technical_termination") for row in rows.values()):
            raise ValueError(f"{label} games.jsonl contains a technical game")
        root_rows = _read_root_audit(run_path)
        root_contract = validate_root_audit(list(rows.values()), root_rows, mode="fixed")
        if root_contract["roots"] != summary.get("root_contract", {}).get("roots"):
            raise ValueError(f"{label} root audit disagrees with its summary")
        if summary.get("learner_samples") != sum(
                len(row["raw_record"]["positions"]) for row in rows.values()):
            raise ValueError(f"{label} learner sample count does not match its games")
        summaries[label] = summary
        rows_by_run[label] = rows
        runs_report[label] = {
            "source_revision": summary["source_revision"],
            "games": summary["games"],
            "roots": root_contract["roots"],
            "learner_samples": summary["learner_samples"],
            "technical_games": summary["technical_games"],
            "inference_batches_size_one": summary["inference_batches"]["all_batches_size_one"],
            "semantic_games_sha256": summary["semantic_games_sha256"],
            "target_field_sha256": summary["target_field_sha256"],
            "path": str(run_path),
        }

    labels = ("A1", "A2", "B", "C")
    comparisons = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            left_rows, right_rows = rows_by_run[left_label], rows_by_run[right_label]
            comparison = _DIAGNOSTICS.compare_game_runs(
                left_rows, right_rows, f"{left_label} vs {right_label}")
            common_ids = set(left_rows) & set(right_rows)
            semantic_hash_mismatches = sum(
                left_rows[game_id]["semantic_sha256"]
                != right_rows[game_id]["semantic_sha256"] for game_id in common_ids)
            target_equal = (
                summaries[left_label]["target_field_sha256"]
                == summaries[right_label]["target_field_sha256"]
                and summaries[left_label]["learner_samples"]
                == summaries[right_label]["learner_samples"])
            exact_games = bool(
                comparison["same_game_id_set"]
                and semantic_hash_mismatches == 0
                and comparison["root_visit_mismatch_count"] == 0
                and comparison["selected_action_mismatch_count"] == 0
                and comparison["action_trace_mismatch_games"] == 0
                and comparison["result_mismatch_games"] == 0
                and comparison["length_mismatch_games"] == 0)
            comparisons.append({**comparison,
                                "semantic_game_hash_mismatch_count": semantic_hash_mismatches,
                                "target_builder_exact": target_equal,
                                "complete_selfplay_exact": exact_games})

    passed = (all(item["complete_selfplay_exact"] and item["target_builder_exact"]
                  for item in comparisons)
              and all(summary["games"] == int(args.games)
                      and summary["technical_games"] == 0
                      and summary["inference_batches"]["all_batches_size_one"]
                      for summary in summaries.values()))
    report = {
        "schema": "torus9-deterministic-core-report-v1",
        "games_per_run": int(args.games), "run_id": FIXED_RUN_ID,
        "master_seed": MASTER_SEED,
        "execution": {label: summaries[label]["execution"] for label in labels},
        "inference_batches_size_one": {
            label: summaries[label]["inference_batches"]["all_batches_size_one"]
            for label in labels},
        "runs": runs_report, "comparisons": comparisons,
        "status": "PASS" if passed else "FAIL",
    }
    write_json(output / "deterministic-core.json", report)
    return report


def _load_compact_games(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            raw = row["raw_record"]
            compact_positions = [{
                "ply": int(position["ply"]),
                "root_visits": [int(value) for value in position["root_visits"]],
                "selected_action": position["selected_action"],
            } for position in raw["positions"]]
            result[str(row["game_id"])] = {
                "game_id": str(row["game_id"]),
                "semantic_sha256": row["semantic_sha256"],
                "raw_record": {"positions": compact_positions,
                               "final_action_trace": list(raw["final_action_trace"]),
                               "formal_result": raw.get("formal_result")},
            }
    return result


def run_concurrent_report(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    a_paths = list(args.a_run)
    if len(a_paths) < 5:
        raise ValueError("production-concurrent baseline requires five revision-A runs")
    revision_runs = {"A": a_paths, "B": [args.b_run], "C": [args.c_run]}
    run_summaries: dict[str, dict[str, Any]] = {}
    for label, paths in revision_runs.items():
        for index, path_text in enumerate(paths):
            path = Path(path_text)
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if (summary["source_revision"] != EXPECTED_REVISIONS[label]
                    or summary["mode"] != "fixed" or int(summary["games"]) != 64
                    or summary["run_id"] != FIXED_RUN_ID
                    or int(summary["master_seed"]) != MASTER_SEED
                    or summary["execution"] != {
                        "workers": 16, "active_games_per_worker": 4,
                        "inference_batch_cap": 64, "inference_batch_wait_ms": 1.0,
                        "start_method": "spawn"}):
                raise ValueError(f"concurrent run has a different identity/config: {path}")
            run_summaries[f"{label}{index + 1}"] = {**summary, "path": str(path)}

    aa_pairs = []
    for left_index in range(len(a_paths)):
        left_path = Path(a_paths[left_index])
        left_rows = _load_compact_games(left_path / "games.jsonl")
        for right_index in range(left_index + 1, len(a_paths)):
            right_path = Path(a_paths[right_index])
            right_rows = _load_compact_games(right_path / "games.jsonl")
            aa_pairs.append(_DIAGNOSTICS.compare_game_runs(
                left_rows, right_rows, f"A{left_index + 1} vs A{right_index + 1}"))
            del right_rows
        del left_rows
    baseline = _DIAGNOSTICS.aggregate_same_revision_baseline(aa_pairs)

    b_rows = _load_compact_games(Path(args.b_run) / "games.jsonl")
    c_rows = _load_compact_games(Path(args.c_run) / "games.jsonl")
    cross = {"A_vs_B": [], "B_vs_C": [], "A_vs_C": []}
    for index, path_text in enumerate(a_paths):
        a_rows = _load_compact_games(Path(path_text) / "games.jsonl")
        ab = _DIAGNOSTICS.compare_game_runs(a_rows, b_rows, f"A{index + 1} vs B")
        ac = _DIAGNOSTICS.compare_game_runs(a_rows, c_rows, f"A{index + 1} vs C")
        cross["A_vs_B"].append({**ab, "baseline_classification":
                                _DIAGNOSTICS.classify_against_baseline(ab, baseline)})
        cross["A_vs_C"].append({**ac, "baseline_classification":
                                _DIAGNOSTICS.classify_against_baseline(ac, baseline)})
        del a_rows
    bc = _DIAGNOSTICS.compare_game_runs(b_rows, c_rows, "B vs C")
    cross["B_vs_C"].append({**bc, "baseline_classification":
                            _DIAGNOSTICS.classify_against_baseline(bc, baseline)})
    baseline_consistent = all(
        comparison["baseline_classification"]["inside_all_observed_AA_ranges"]
        for comparisons in cross.values() for comparison in comparisons)
    summary = {
        "schema": "torus9-production-concurrent-baseline-v1",
        "game_count_per_run": 64, "same_revision_A_run_count": len(a_paths),
        "same_revision_A_pair_count": len(aa_pairs),
        "same_revision_baseline": baseline, "cross_revision": cross,
        "cross_revision_inside_AA_observed_ranges": baseline_consistent,
        "run_identities": run_summaries,
        "strict_production_bitwise_reproducible": baseline["all_pairs_bitwise_identical"],
        "interpretation": "descriptive empirical ranges; no significance or arbitrary tolerance threshold",
    }
    write_json(output, summary)
    return summary


def _load_game_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["game_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate game ID in {path}")
    return result


def compare_games(left_path: Path, right_path: Path) -> dict[str, Any]:
    left, right = _load_game_rows(left_path), _load_game_rows(right_path)
    if set(left) != set(right):
        return {"identical": False, "identical_games": 0,
                "left_games": len(left), "right_games": len(right),
                "missing_left": sorted(set(right) - set(left)),
                "missing_right": sorted(set(left) - set(right)),
                "first_divergence": None}
    matches = 0
    first = None
    for game_id in sorted(left):
        if left[game_id]["semantic_sha256"] == right[game_id]["semantic_sha256"]:
            matches += 1
            continue
        if first is None:
            l_sem, r_sem = left[game_id]["semantic"], right[game_id]["semantic"]
            index = 0
            for index, (lp, rp) in enumerate(zip(l_sem["positions"], r_sem["positions"])):
                if lp != rp:
                    break
            common = min(len(l_sem["positions"]), len(r_sem["positions"]))
            if index >= common:
                index = max(0, common - 1)
            lp = l_sem["positions"][index] if l_sem["positions"] else None
            rp = r_sem["positions"][index] if r_sem["positions"] else None
            first = {
                "game_id": game_id,
                "game_semantic_diff": first_mapping_difference(l_sem, r_sem),
                "ply_index": index + 1,
                "left_root": lp,
                "right_root": rp,
                "left_raw_record": left[game_id]["raw_record"],
                "right_raw_record": right[game_id]["raw_record"],
            }
    return {"identical": matches == len(left), "identical_games": matches,
            "left_games": len(left), "right_games": len(right),
            "first_divergence": first}


def compare_targets(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    if left == right:
        return {"identical": True, "sample_count": left["sample_count"],
                "field_sha256": left["field_sha256"],
                "sample_identity_sha256": left["sample_identity_sha256"]}
    return {"identical": False,
            "first_difference": first_mapping_difference(left, right),
            "left_sample_count": left.get("sample_count"),
            "right_sample_count": right.get("sample_count")}


def _compare_target_summaries(left_path: Path, right_path: Path,
                              label: str) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_semantic = {key: value for key, value in left.items() if key != "source_revision"}
    right_semantic = {key: value for key, value in right.items() if key != "source_revision"}
    return {
        "comparison": label,
        "identical": left_semantic == right_semantic,
        "first_difference": (None if left_semantic == right_semantic
                              else first_mapping_difference(left_semantic, right_semantic)),
        "sample_count": left.get("learner_samples"),
        "field_sha256": left.get("field_sha256"),
        "sample_identity_sha256": left.get("sample_identity_sha256"),
        "left_revision": left.get("source_revision"),
        "right_revision": right.get("source_revision"),
    }


def _compare_training(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    ignored = {"source_revision"}
    left_scientific = {key: value for key, value in left.items() if key not in ignored}
    right_scientific = {key: value for key, value in right.items() if key not in ignored}
    if left_scientific == right_scientific:
        return {"identical": True, "updates": left["updates"],
                "sample_rows_per_update": 64, "model_hash_final": left["model_hash_final"],
                "adam_state_sha256": left["final_adam_state"]["sha256"],
                "dataset_sha256": left["dataset_sha256"]}
    return {"identical": False,
            "first_difference": first_mapping_difference(left_scientific, right_scientific)}


def _compare_integration(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_rows = {row["game_position_identity"]: row for row in left["row_request_mapping"]}
    right_rows = {row["game_position_identity"]: row for row in right["row_request_mapping"]}
    left_roots = {row["identity"]: row for row in left["mcts_roots"]}
    right_roots = {row["identity"]: row for row in right["mcts_roots"]}
    result = {
        "identical_positions": set(left_rows) == set(right_rows) == set(left_roots) == set(right_roots),
        "positions": len(left_rows),
        "lost_or_misrouted_rows": int(left.get("lost_or_misrouted_rows", 1))
                                  + int(right.get("lost_or_misrouted_rows", 1)),
        "strict_policy_wdl_equal": True,
        "root_mcts_equal": True,
        "recorded_root_replay": {
            "B_matching_roots": sum(bool(row.get("matches_recorded_root")) for row in left["mcts_roots"]),
            "C_matching_roots": sum(bool(row.get("matches_recorded_root")) for row in right["mcts_roots"]),
            "roots_per_revision": len(left["mcts_roots"]),
            "same_match_status_by_position": all(
                bool(left_roots[key].get("matches_recorded_root"))
                == bool(right_roots[key].get("matches_recorded_root"))
                for key in set(left_roots) & set(right_roots)),
        },
    }
    if set(left_rows) != set(right_rows) or set(left_roots) != set(right_roots):
        result.update(identical_positions=False, strict_policy_wdl_equal=False, root_mcts_equal=False)
        return result
    for identity in sorted(left_rows):
        left_sem = {key: value for key, value in left_rows[identity].items()
                    if key not in {"request_id", "worker_id", "lane_id", "row", "slot_id"}}
        right_sem = {key: value for key, value in right_rows[identity].items()
                     if key not in {"request_id", "worker_id", "lane_id", "row", "slot_id"}}
        if left_sem != right_sem:
            result["strict_policy_wdl_equal"] = False
            result.setdefault("first_mapping_difference", first_mapping_difference(left_sem, right_sem))
        if left_roots[identity] != right_roots[identity]:
            result["root_mcts_equal"] = False
            result.setdefault("first_root_difference",
                              first_mapping_difference(left_roots[identity], right_roots[identity]))
    return result


def _source_status(repo: Path, expected: str) -> dict[str, Any]:
    revision = _revision(repo)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True).strip()
    if revision != expected:
        raise ValueError(f"Revision {repo} is {revision}, expected {expected}")
    if status:
        raise ValueError(f"Immutable source worktree is dirty: {repo}: {status[:500]}")
    return {"revision": revision, "clean": True}


def _checkpoint_safety(checkpoint: Path, checkpoint_ref_path: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    lineage = checkpoint.parent.parent
    manifest = lineage / "manifest.json"
    metadata = checkpoint.with_suffix(".metadata.json")
    node_path = checkpoint_ref_path.resolve()
    node = json.loads(node_path.read_text(encoding="utf-8"))
    checkpoint_ref = node.get("checkpoint")
    if not isinstance(checkpoint_ref, Mapping):
        raise ValueError("M233 artifact graph node has no checkpoint reference")
    expected_sha = str(checkpoint_ref["sha256"]).removeprefix("sha256:")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError("Canonical M233 reference SHA does not match the checkpoint file")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if str(meta.get("artifact_sha256", "")).removeprefix("sha256:") != actual_sha:
        raise ValueError("M233 sidecar metadata SHA does not match the checkpoint")
    provenance_path = lineage / str(node["provenance"]["path"])
    config_artifact = node["effective_config"]["artifact"]
    config_path = lineage / str(config_artifact["path"])
    if sha256_file(config_path) != str(config_artifact["sha256"]).removeprefix("sha256:"):
        raise ValueError("M233 effective-config artifact SHA does not match its graph node")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_context = {
        "topology": "torus9",
        "compatibility": {"input_channels": 5, "observation_shape": [5, 81]},
        "self_play": {"komi": 1.5, "games_per_iteration": 768,
                      "mcts_simulations": 200, "cpuct": 1.25, "fpu": 0.0,
                      "root_noise": True, "dirichlet_epsilon": 0.25,
                      "dirichlet_alpha": 0.11,
                      "temperature": "1.0 on plies 1–8, then 0", "resign": False},
        "training": {"optimizer": "Adam", "batch_size": 64,
                     "learning_rate": 1e-4, "optimizer_steps_per_iteration": 1280,
                     "gradient_clip": 8.0, "weight_decay": 0.0, "l2_sp": False},
        "replay": {"generations": 3, "cap": None},
        "execution": {"device": "cuda", "workers": 16,
                      "active_games_per_worker": 4, "inference_batch_cap": 64,
                      "inference_batch_wait_ms": 1.0,
                      "selfplay_master_seed": MASTER_SEED,
                      "training_master_seed": 2026092701},
    }
    for section, expected in expected_context.items():
        actual = config.get(section)
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or any(actual.get(key) != value
                                                       for key, value in expected.items()):
                raise ValueError(f"M233 effective scientific context drift in {section}")
        elif actual != expected:
            raise ValueError(f"M233 effective scientific context drift in {section}")
    m234 = checkpoint.parent / "M234.pt"
    files = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "checkpoint_metadata": str(metadata),
        "checkpoint_metadata_sha256": sha256_file(metadata),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "artifact_node": str(node_path),
        "artifact_node_sha256": sha256_file(node_path),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "effective_config": str(config_path),
        "effective_config_sha256": sha256_file(config_path),
        "m234_checkpoint_path": str(m234),
        "m234_exists_before": m234.exists(),
    }
    return {"files": files, "checkpoint_ref": dict(checkpoint_ref),
            "checkpoint_metadata": meta,
            "scientific_context": expected_context}


def _check_safety_unchanged(before: Mapping[str, Any], checkpoint_ref_path: Path) -> dict[str, Any]:
    paths = before["files"]
    after = {
        "checkpoint_sha256": sha256_file(Path(paths["checkpoint"])),
        "checkpoint_metadata_sha256": sha256_file(Path(paths["checkpoint_metadata"])),
        "manifest_sha256": sha256_file(Path(paths["manifest"])),
        "artifact_node_sha256": sha256_file(Path(paths["artifact_node"])),
        "provenance_sha256": sha256_file(Path(paths["provenance"])),
        "effective_config_sha256": sha256_file(Path(paths["effective_config"])),
        "m234_exists_after": Path(paths["m234_checkpoint_path"]).exists(),
    }
    expected = {"checkpoint_sha256": paths["checkpoint_sha256"],
                "checkpoint_metadata_sha256": paths["checkpoint_metadata_sha256"],
                "manifest_sha256": paths["manifest_sha256"],
                "artifact_node_sha256": paths["artifact_node_sha256"],
                "provenance_sha256": paths["provenance_sha256"],
                "effective_config_sha256": paths["effective_config_sha256"],
                "m234_exists_after": paths["m234_exists_before"]}
    if after != expected:
        raise RuntimeError(f"M233 artifact safety check changed: before={expected}, after={after}")
    return {"unchanged": True, "before": paths, "after": after,
            "production_replay_or_lineage_writes": "NO (all worker output roots were under /tmp)"}


def _launch(python: str, script: Path, mode: str, args: Sequence[object]) -> None:
    command = [python, str(script), mode, *[str(arg) for arg in args]]
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.run(command, check=True, env=env)


def _launch_selfplay(python: str, script: Path, *, mode: str, repo: Path,
                     expected_revision: str, checkpoint: Path, output: Path,
                     expected_checkpoint_sha: str, games: int, run_id: str,
                     workers: int, active_games_per_worker: int,
                     checkpoint_ref: Path | None = None) -> None:
    argv: list[object] = [
        "--repo", repo, "--expected-revision", expected_revision,
        "--checkpoint", checkpoint, "--expected-checkpoint-sha", expected_checkpoint_sha,
        "--output", output, "--games", games, "--run-id", run_id,
        "--seed", MASTER_SEED, "--workers", workers,
        "--active-games-per-worker", active_games_per_worker, "--device", "cuda",
    ]
    if mode == "pcr":
        if checkpoint_ref is None:
            raise ValueError("PCR resume audit requires the canonical M233 artifact node")
        argv.extend(["--save-shard", "--checkpoint-ref", checkpoint_ref])
    _launch(python, script, "worker-selfplay", ["--mode", mode, *argv])


def _comparison_report(left: Mapping[str, Any], right: Mapping[str, Any],
                       label: str) -> dict[str, Any]:
    report = compare_games(Path(left["games_path"]), Path(right["games_path"]))
    report["comparison"] = label
    return report


def compare_repeat_root_targets(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Summarize repeated fixed-mode root-target determinism for one revision."""
    left, right = _load_game_rows(left_path), _load_game_rows(right_path)
    common = sorted(set(left) & set(right))
    mismatch_games: set[str] = set()
    mismatch_positions = 0
    trace_equal = winner_equal = length_equal = True
    first_divergence = None
    for game_id in common:
        lraw, rraw = left[game_id]["raw_record"], right[game_id]["raw_record"]
        lpositions, rpositions = lraw["positions"], rraw["positions"]
        lsemantic = left[game_id]["semantic"]["positions"]
        rsemantic = right[game_id]["semantic"]["positions"]
        trace_equal &= lraw["final_action_trace"] == rraw["final_action_trace"]
        winner_equal &= lraw.get("formal_result") == rraw.get("formal_result")
        length_equal &= len(lpositions) == len(rpositions)
        for index, (lp, rp) in enumerate(zip(lpositions, rpositions)):
            if lp["root_visits"] == rp["root_visits"] and lp["pi"] == rp["pi"]:
                continue
            mismatch_positions += 1
            mismatch_games.add(game_id)
            if first_divergence is None:
                first_divergence = {
                    "game_id": game_id, "ply_index": index + 1,
                    "state_sha256": sha256_json(lp["state"]),
                    "legal_action_identity": {
                        "left": lsemantic[index]["legal_action_identity"],
                        "right": rsemantic[index]["legal_action_identity"],
                    },
                    "rng_identity": {
                        "left_game_seed": int(lraw["game_seed"]),
                        "right_game_seed": int(rraw["game_seed"]),
                        "left_search_seed": int(lp["search_seed"]),
                        "right_search_seed": int(rp["search_seed"]),
                    },
                    "left_root_visits": [int(value) for value in lp["root_visits"]],
                    "right_root_visits": [int(value) for value in rp["root_visits"]],
                    "changed_action_indices": [
                        {"action_index": action_index, "left": int(lvalue), "right": int(rvalue)}
                        for action_index, (lvalue, rvalue) in enumerate(
                            zip(lp["root_visits"], rp["root_visits"])) if lvalue != rvalue],
                    "left_selected_action": lp["selected_action"],
                    "right_selected_action": rp["selected_action"],
                    "left_raw_record": lraw,
                    "right_raw_record": rraw,
                }
    identical_games = sum(
        left[game_id]["semantic_sha256"] == right[game_id]["semantic_sha256"]
        for game_id in common)
    same_game_set = set(left) == set(right)
    return {
        "identical": same_game_set and identical_games == len(left) == len(right),
        "games": len(left), "right_games": len(right),
        "identical_games": identical_games,
        "root_visit_mismatch_games": len(mismatch_games),
        "root_visit_mismatch_positions": mismatch_positions,
        "action_trace_winner_and_length_equal": trace_equal and winner_equal and length_equal,
        "action_traces_equal": trace_equal,
        "winners_equal": winner_equal,
        "ply_lengths_equal": length_equal,
        "same_game_id_set": same_game_set,
        "first_divergence": first_divergence,
    }


def _write_first_divergence_diagnostic(*, comparison: Mapping[str, Any],
                                       left_repo: Path, left_revision: str,
                                       right_repo: Path, right_revision: str,
                                       checkpoint: Path, expected_checkpoint_sha: str,
                                       output: Path, python: str, script: Path) -> dict[str, Any] | None:
    divergence = comparison.get("first_divergence")
    if not divergence:
        return None
    left_record = divergence["left_raw_record"]
    right_record = divergence["right_raw_record"]
    index = int(divergence["ply_index"]) - 1
    if not left_record.get("positions") or not right_record.get("positions"):
        return {"diagnostic_unavailable": "one side has no recorded root positions",
                "first_divergence": divergence.get("game_semantic_diff")}
    lp = left_record["positions"][min(index, len(left_record["positions"]) - 1)]
    rp = right_record["positions"][min(index, len(right_record["positions"]) - 1)]
    diagnostic_root = output / "first-divergence"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    state_path = diagnostic_root / "common-left-state.json"
    left_root_path = diagnostic_root / "left-root.json"
    right_root_path = diagnostic_root / "right-root.json"
    write_json(state_path, lp["state"])
    write_json(left_root_path, {
        "game_id": left_record["game_id"], "ply": lp["ply"],
        "state_identity": lp["state"], "root_visits": lp["root_visits"],
        "selected_action": lp["selected_action"], "search_seed": lp["search_seed"],
        "game_seed": left_record["game_seed"],
        "policy": lp.get("pi"), "search_mode": lp.get("search_mode", "fixed"),
    })
    write_json(right_root_path, {
        "game_id": right_record["game_id"], "ply": rp["ply"],
        "state_identity": rp["state"], "root_visits": rp["root_visits"],
        "selected_action": rp["selected_action"], "search_seed": rp["search_seed"],
        "game_seed": right_record["game_seed"],
        "policy": rp.get("pi"), "search_mode": rp.get("search_mode", "fixed"),
    })
    outputs = []
    for side, repo, revision in (("left", left_repo, left_revision),
                                 ("right", right_repo, right_revision)):
        side_output = diagnostic_root / side
        _launch(python, script, "worker-diagnostic", [
            "--repo", repo, "--expected-revision", revision,
            "--checkpoint", checkpoint, "--expected-checkpoint-sha", expected_checkpoint_sha,
            "--output", side_output, "--state-json", state_path,
            "--master-seed", MASTER_SEED, "--game-seed", left_record["game_seed"],
            "--ply", lp["ply"], "--search-seed", lp["search_seed"],
            "--left-root-json", left_root_path, "--right-root-json", right_root_path,
        ])
        outputs.append(str(side_output / "first-divergence.json"))
    result = {"output_files": outputs,
              "same_input_state_sha256": sha256_json(lp["state"]),
              "left_state_sha256": sha256_json(lp["state"]),
              "right_state_sha256": sha256_json(rp["state"]),
              "left_root_visits_sha256": sha256_json(lp["root_visits"]),
              "right_root_visits_sha256": sha256_json(rp["root_visits"]),
              "left_selected_action": lp["selected_action"],
              "right_selected_action": rp["selected_action"],
              "rng_identity": {"master_seed": left_record["master_seed"],
                               "game_seed": left_record["game_seed"],
                               "ply": lp["ply"], "search_seed": lp["search_seed"]}}
    diagnostic_rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in outputs]
    result["direct_network_policy_sha256"] = [row["network_policy_sha256"]
                                               for row in diagnostic_rows]
    result["direct_network_wdl_sha256"] = [row["network_wdl_sha256"]
                                             for row in diagnostic_rows]
    result["direct_network_policy_wdl_equal"] = all(
        row["network_policy_sha256"] == diagnostic_rows[0]["network_policy_sha256"]
        and row["network_wdl_sha256"] == diagnostic_rows[0]["network_wdl_sha256"]
        for row in diagnostic_rows)
    write_json(diagnostic_root / "first-divergence-summary.json", result)
    return result


def _report_markdown(summary: Mapping[str, Any]) -> str:
    revisions = summary.get("revisions", {})
    fixed = summary.get("fixed_parity", {})
    targets = summary.get("fixed_targets", {})
    optimizer = summary.get("optimizer", {})
    pcr = summary.get("pcr", {})
    safety = summary.get("artifact_safety", {})
    lines = [
        "# Scientific Continuation Gate",
        "",
        "## Revisions",
        "",
        f"- A: `{revisions.get('A', {}).get('revision', EXPECTED_REVISIONS['A'])}` (before PR #230)",
        f"- B: `{revisions.get('B', {}).get('revision', EXPECTED_REVISIONS['B'])}` (after PR #230, before PR #231)",
        f"- C: `{revisions.get('C', {}).get('revision', EXPECTED_REVISIONS['C'])}` (current main after PR #231)",
        f"- M233 checkpoint: `{summary.get('m233', {}).get('files', {}).get('checkpoint', '')}`",
        f"- M233 SHA-256: `{summary.get('m233', {}).get('files', {}).get('checkpoint_sha256', '')}`",
        f"- M233 model hash: `{summary.get('m233_model_hash', '')}`",
        "",
        "## Fixed self-play parity",
        "",
    ]
    for label in ("A_vs_B", "B_vs_C", "A_vs_C"):
        item = fixed.get(label, {})
        lines.append(f"- {label.replace('_', ' ')}: {item.get('identical_games', 0)}/{item.get('left_games', 64)} semantic games identical")
    lines.extend([
        f"- Games per revision: {summary.get('fixed_games', 0)}",
        f"- Complete-game plies per revision: {summary.get('fixed_plies', 0)}",
        f"- MCTS roots per revision: {summary.get('fixed_roots', 0)}",
        f"- Technical games across A/B/C: {summary.get('fixed_technical_games', 'unverified')}",
        "",
        "## Central inference and MCTS integration",
        "",
        f"- B vs C output row/request mapping: {'PASS' if summary.get('central_inference', {}).get('lost_or_misrouted_rows') == 0 else 'FAIL'}",
        f"- Strict policy/WDL parity: {'PASS' if summary.get('central_inference', {}).get('strict_policy_wdl_equal') else 'FAIL'}",
        f"- B/C isolated MCTS root distributions and selected moves: {'PASS' if summary.get('central_inference', {}).get('root_mcts_equal') else 'FAIL'}",
        f"- Isolated replay roots matching recorded full-run roots: B {summary.get('central_inference', {}).get('recorded_root_replay', {}).get('B_matching_roots', 0)}/{summary.get('central_inference', {}).get('recorded_root_replay', {}).get('roots_per_revision', 0)}; C {summary.get('central_inference', {}).get('recorded_root_replay', {}).get('C_matching_roots', 0)}/{summary.get('central_inference', {}).get('recorded_root_replay', {}).get('roots_per_revision', 0)} (diagnostic; end-to-end roots are gated by complete-game parity)",
        f"- Positions: {summary.get('central_inference', {}).get('positions', 0)}",
        "",
        "## Fixed learner targets",
        "",
        f"- Shared raw-game source: revision {targets.get('raw_games_source_revision', 'unverified')}, "
        f"{targets.get('raw_games_count', 0)} complete games",
        f"- A/B/C target-builder identity on that same raw data: {'PASS' if targets.get('identical') else 'FAIL'}",
        f"- Targets generated inside each revision's own self-play run: "
        f"{'identical' if targets.get('selfplay_embedded_targets', {}).get('identical') else 'differ where their raw self-play results differ'}",
        f"- Learner samples: {targets.get('sample_count', 0)}",
    ])
    for field_name, digest in targets.get("field_sha256", {}).items():
        lines.append(f"- `{field_name}` tensor SHA-256: `{digest}`")
    lines.extend([
        "",
        "## Bounded optimizer continuation",
        "",
        f"- Adam updates: {optimizer.get('updates', 0)}",
        f"- Sampled row identities and loss/gradient reports: {'identical' if optimizer.get('identical') else 'mismatch'}",
        f"- Final model tensor hash: `{optimizer.get('model_hash_final', '')}`",
        f"- Final Adam state hash: `{optimizer.get('adam_state_sha256', '')}`",
        "",
        "## PCR invariants, order, and resume",
        "",
        f"- Completed games per execution: {pcr.get('games', 0)}",
        f"- Both worker configurations completed the same game IDs: {'PASS' if pcr.get('runs_complete') and pcr.get('same_game_ids_across_orderings') else 'FAIL'}",
        f"- Root count: {pcr.get('roots', 0)}; cheap: {pcr.get('cheap_roots', 0)}; full: {pcr.get('full_roots', 0)}",
        f"- Alternate worker-order roots: {pcr.get('invariants_by_execution', {}).get('worker_order_workers_16', {}).get('roots', 'unverified')}; "
        f"cheap: {pcr.get('invariants_by_execution', {}).get('worker_order_workers_16', {}).get('cheap_roots', 'unverified')}; "
        f"full: {pcr.get('invariants_by_execution', {}).get('worker_order_workers_16', {}).get('full_roots', 'unverified')}",
        f"- Invalid cap/noise/eligibility combinations: {pcr.get('invalid_cap_noise_eligibility_combinations', 'unverified')}",
        f"- Cheap learner leakage: {pcr.get('cheap_learner_leakage', 'unverified')}",
        f"- Missing full learner positions: {pcr.get('missing_full_learner_positions', 'unverified')}",
        f"- Terminal WDL/ownership/score targets: {'PASS' if pcr.get('terminal_targets') else 'FAIL'}",
        f"- Recorded PCR choices match versioned `(game_seed, ply)` derivation in both runs: "
        f"{'PASS' if all(run.get('selection_matches_game_ply_seed') for run in pcr.get('invariants_by_execution', {}).values()) else 'FAIL'}",
        f"- Worker-order decision identity on shared `(game_id, ply)` positions: {'PASS' if pcr.get('worker_order_identity') else 'FAIL'}",
        f"- Shared decisions: {pcr.get('worker_order_comparison', {}).get('shared_positions', 0)}; "
        f"unmatched positions after trajectory differences: left {pcr.get('worker_order_comparison', {}).get('left_only_positions', 0)}, "
        f"right {pcr.get('worker_order_comparison', {}).get('right_only_positions', 0)}",
        f"- Committed test shard read-only reload and semantic identity: {'PASS' if pcr.get('committed_shard_load_identity') else 'FAIL'}",
        f"- Isolated production `run_generation` interruption/resume vs uninterrupted fixture "
        f"(self-play mocked; pytest temporary lineage): {'PASS' if pcr.get('production_resume_test_passed') else 'FAIL'}",
        f"- Uninterrupted vs reloaded raw semantic games: {'PASS' if pcr.get('resume_semantic_matches_uninterrupted') else 'FAIL'}",
        f"- Uninterrupted vs reloaded learner target summary: {'PASS' if pcr.get('resume_target_matches_uninterrupted') else 'FAIL'}",
        "",
        "## Artifact safety",
        "",
        f"- M233 checkpoint and lineage manifest unchanged: {'YES' if safety.get('unchanged') else 'NO'}",
        f"- M234 created: {'NO' if not safety.get('after', {}).get('m234_exists_after') else 'already present; unchanged'}",
        f"- Production replay or lineage writes by this audit: {safety.get('production_replay_or_lineage_writes', 'unverified')}",
        "",
        "## Evidence statement",
        "",
        "PR #231 already established direct M233 policy/WDL bitwise parity at batch sizes 1, 22, and 64.",
        "This audit adds bounded evidence through the shared inference dispatcher, MCTS, complete self-play, learner targets, and a 32-update continuation; PCR is checked separately against its declared invariants.",
        "",
        ("No semantic divergence was observed across the bounded deterministic continuation gate; fixed-mode self-play and learner data remained identical across the tested revisions, and PCR-specific invariants passed."
         if summary.get("verdict") == "PASS" else
         "The gate did not pass. The measured divergence, failed invariant, or execution limitation is recorded below; no scientific continuation claim follows from this run."),
        "",
        "## Final verdict",
        "",
        f"`SCIENTIFIC CONTINUATION GATE: {summary.get('verdict', 'INCOMPLETE')}`",
        "",
    ])
    error = summary.get("error")
    if error:
        lines.extend(["## Failure or limitation", "", str(error), ""])
    diagnostic = summary.get("first_divergence_diagnostic")
    if diagnostic:
        lines.extend(["## First cross-revision divergence", ""])
        comparison = fixed.get("A_vs_B", {}).get("first_divergence")
        if comparison:
            left_root, right_root = comparison.get("left_root", {}), comparison.get("right_root", {})
            left_visits, right_visits = left_root.get("root_visits", []), right_root.get("root_visits", [])
            visit_differences = [
                {"action_index": index, "A": left, "B": right}
                for index, (left, right) in enumerate(zip(left_visits, right_visits)) if left != right]
            lines.extend([
                f"- Game/ply: `{comparison.get('game_id')}` / {comparison.get('ply_index')}",
                f"- State SHA-256: `{left_root.get('state_sha256', '')}`; legal mask SHA-256: "
                f"`{left_root.get('legal_action_identity', {}).get('mask_sha256', '')}`",
                f"- Search seeds: game `{left_root.get('rng_identity', {}).get('game_seed')}`, "
                f"search `{left_root.get('rng_identity', {}).get('search_seed')}`",
                f"- Root visit differences (action index, A → B): `{visit_differences}`",
                f"- Selected action: A `{left_root.get('selected_action')}`, B `{right_root.get('selected_action')}`",
                f"- Direct policy/WDL replay on the same state: "
                f"{'bitwise equal' if diagnostic.get('direct_network_policy_wdl_equal') else 'different or unverified'}; "
                f"policy hashes `{diagnostic.get('direct_network_policy_sha256', [])}`, "
                f"WDL hashes `{diagnostic.get('direct_network_wdl_sha256', [])}`",
                "",
            ])
        lines.extend(["Diagnostic record:", "", "```json",
                      json.dumps(diagnostic, sort_keys=True, indent=2), "```", ""])
    repeat = summary.get("same_revision_repeat")
    if repeat:
        root_difference = repeat.get("first_root_difference") or {}
        lines.extend(["## Same-revision repeat diagnostic", "",
                      f"- Revision A repeated with the same checkpoint, game IDs, master seed, and 16-worker configuration: "
                      f"{repeat.get('identical_games', 0)}/{repeat.get('games', 0)} complete-game semantic identities matched.",
                      f"- Root visit differences: {repeat.get('root_visit_mismatch_positions', 'unverified')} positions "
                      f"across {repeat.get('root_visit_mismatch_games', 'unverified')} games.",
                      f"- Action traces / winners / plies unchanged: {'YES' if repeat.get('action_trace_winner_and_length_equal') else 'NO'}.",
                      f"- First root difference: game `{root_difference.get('game_id', 'unavailable')}`, "
                      f"ply {root_difference.get('ply_index', 'unavailable')}; state SHA-256 "
                      f"`{root_difference.get('state_sha256', '')}`; legal mask SHA-256 "
                      f"`{root_difference.get('legal_action_identity', {}).get('left', {}).get('mask_sha256', '')}`.",
                      f"- Root visits differ at `{root_difference.get('changed_action_indices', [])}`; "
                      f"selected actions {root_difference.get('left_selected_action')} and "
                      f"{root_difference.get('right_selected_action')}; RNG `{root_difference.get('rng_identity', {})}`.",
                      f"- Direct M233 policy/WDL replay on the divergent state: {'bitwise equal' if repeat.get('direct_network_policy_wdl_equal') else 'not established'}.",
                      f"- Direct policy hash `{repeat.get('direct_network_policy_sha256', '')}`; "
                      f"WDL hash `{repeat.get('direct_network_wdl_sha256', '')}`.",
                      "- The recorded root-target mismatch is in the concurrent full-self-play path. This audit does not prove which concurrent search event causes the visit allocation to differ.",
                      ""])
    return "\n".join(lines)


def _run_focused_tests(python: str, repo: Path, output: Path) -> dict[str, Any]:
    log_path = output / "focused-tests.log"
    tests_root = Path(__file__).resolve().parents[1]
    command = [python, "-m", "pytest", "-q",
               str(tests_root / "test_scientific_continuation_gate.py"),
               str(tests_root / "test_torus9_pcr.py"),
               "tests/test_central_inference_batch_output.py"]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False, env=dict(os.environ))
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
    result = {"passed": completed.returncode == 0, "returncode": completed.returncode,
              "command": command, "log": str(log_path), "tail": tail}
    if completed.returncode:
        raise RuntimeError(f"Focused PCR/central-inference tests failed; see {log_path}:\n"
                           + "\n".join(tail))
    return result


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_tmp_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Audit output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_ref = (Path(args.checkpoint_ref).resolve() if args.checkpoint_ref else
                      checkpoint.parent.parent / "metadata" / "checkpoints" / "M233.json")
    before = _checkpoint_safety(checkpoint, checkpoint_ref)
    if before["files"]["checkpoint_sha256"] != args.expected_checkpoint_sha:
        raise ValueError("Canonical M233 SHA does not match requested expected SHA")
    if args.fixed_games != 64:
        raise ValueError("Full Scientific Continuation Gate requires exactly 64 fixed-mode games")
    if not 16 <= args.pcr_games <= 32:
        raise ValueError("PCR mechanics audit must use 16 to 32 complete games")

    script = Path(__file__).resolve()
    python = args.python
    repos = {"A": Path(args.repo_a).resolve(),
             "B": Path(args.repo_b).resolve(),
             "C": Path(args.repo_c).resolve()}
    summary: dict[str, Any] = {
        "gate": "Scientific Continuation Gate",
        "status": "RUNNING", "verdict": "INCOMPLETE",
        "expected_revisions": dict(EXPECTED_REVISIONS),
        "m233": before, "output_directory": str(output),
        "fixed_games": args.fixed_games, "pcr_games": args.pcr_games,
        "stages": {},
    }
    try:
        summary["revisions"] = {
            name: _source_status(repo, EXPECTED_REVISIONS[name])
            for name, repo in repos.items()
        }
        summary["m233_model_hash"] = str(before["checkpoint_metadata"].get("model_hash", ""))
        checkpoint_sha = before["files"]["checkpoint_sha256"]

        fixed = {}
        for label in ("A", "B", "C"):
            worker_output = output / f"fixed-{label}"
            _launch_selfplay(
                python, script, mode="fixed", repo=repos[label],
                expected_revision=EXPECTED_REVISIONS[label], checkpoint=checkpoint,
                output=worker_output, expected_checkpoint_sha=checkpoint_sha,
                games=args.fixed_games, run_id=FIXED_RUN_ID, workers=16,
                active_games_per_worker=4)
            worker_summary = json.loads((worker_output / "summary.json").read_text(encoding="utf-8"))
            worker_summary["games_path"] = str(worker_output / "games.jsonl")
            worker_summary["targets_path"] = str(worker_output / "targets.json")
            worker_summary["dataset_path"] = str(worker_output / "dataset.pt")
            fixed[label] = worker_summary
        repeat_output = output / "fixed-A-repeat"
        _launch_selfplay(
            python, script, mode="fixed", repo=repos["A"],
            expected_revision=EXPECTED_REVISIONS["A"], checkpoint=checkpoint,
            output=repeat_output, expected_checkpoint_sha=checkpoint_sha,
            games=args.fixed_games, run_id=FIXED_RUN_ID, workers=16,
            active_games_per_worker=4)
        repeat = compare_repeat_root_targets(
            Path(fixed["A"]["games_path"]), repeat_output / "games.jsonl")
        repeat["first_root_difference"] = None
        if repeat.get("first_divergence"):
            divergence = repeat["first_divergence"]
            repeat["first_root_difference"] = {
                key: divergence[key] for key in (
                    "game_id", "ply_index", "state_sha256", "legal_action_identity",
                    "rng_identity", "changed_action_indices", "left_selected_action",
                    "right_selected_action")}
            direct = _write_first_divergence_diagnostic(
                comparison={"first_divergence": divergence},
                left_repo=repos["A"], left_revision=EXPECTED_REVISIONS["A"],
                right_repo=repos["A"], right_revision=EXPECTED_REVISIONS["A"],
                checkpoint=checkpoint, expected_checkpoint_sha=checkpoint_sha,
                output=output / "same-revision-repeat-diagnostic",
                python=python, script=script)
            direct_rows = [json.loads(Path(path).read_text(encoding="utf-8"))
                           for path in direct["output_files"]]
            repeat["direct_network_policy_wdl_equal"] = all(
                row["network_policy_sha256"] == direct_rows[0]["network_policy_sha256"]
                and row["network_wdl_sha256"] == direct_rows[0]["network_wdl_sha256"]
                for row in direct_rows)
            repeat["direct_network_policy_sha256"] = direct_rows[0]["network_policy_sha256"]
            repeat["direct_network_wdl_sha256"] = direct_rows[0]["network_wdl_sha256"]
            repeat["diagnostic"] = direct
        else:
            repeat["direct_network_policy_wdl_equal"] = None
        summary["same_revision_repeat"] = repeat
        summary["same_revision_repeat"]["output_directory"] = str(repeat_output)
        summary["stages"]["fixed_repeatability"] = bool(repeat["identical"])

        fixed_parity = {
            "A_vs_B": _comparison_report(fixed["A"], fixed["B"], "A vs B"),
            "B_vs_C": _comparison_report(fixed["B"], fixed["C"], "B vs C"),
            "A_vs_C": _comparison_report(fixed["A"], fixed["C"], "A vs C"),
        }
        summary["fixed_parity"] = fixed_parity
        summary["fixed_plies"] = fixed["C"]["plies"]
        summary["fixed_roots"] = fixed["C"]["root_contract"]["roots"]
        summary["fixed_technical_games"] = sum(fixed[name]["technical_games"] for name in fixed)
        summary["stages"]["fixed_selfplay"] = all(
            fixed_parity[key]["identical"] and fixed_parity[key]["identical_games"] == 64
            for key in fixed_parity)
        if not summary["stages"]["fixed_selfplay"]:
            for pair, comparison in fixed_parity.items():
                if not comparison["identical"] and comparison.get("first_divergence"):
                    labels = pair.split("_vs_")
                    diagnostic = _write_first_divergence_diagnostic(
                        comparison=comparison, left_repo=repos[labels[0]],
                        left_revision=EXPECTED_REVISIONS[labels[0]],
                        right_repo=repos[labels[1]], right_revision=EXPECTED_REVISIONS[labels[1]],
                        checkpoint=checkpoint, expected_checkpoint_sha=checkpoint_sha,
                        output=output, python=python, script=script)
                    summary["first_divergence_diagnostic"] = diagnostic
                    break

        selfplay_target_pairs = [
            compare_targets(Path(fixed["A"]["targets_path"]), Path(fixed["B"]["targets_path"])),
            compare_targets(Path(fixed["B"]["targets_path"]), Path(fixed["C"]["targets_path"])),
            compare_targets(Path(fixed["A"]["targets_path"]), Path(fixed["C"]["targets_path"])),
        ]
        shared_raw_games = Path(fixed["A"]["games_path"])
        shared_targets = {}
        for label in ("A", "B", "C"):
            target_output = output / f"fixed-targets-from-A-{label}"
            _launch(python, script, "worker-targets", [
                "--repo", repos[label], "--expected-revision", EXPECTED_REVISIONS[label],
                "--games-jsonl", shared_raw_games, "--output", target_output,
            ])
            shared_targets[label] = target_output / "summary.json"
        target_pairs = [
            _compare_target_summaries(shared_targets["A"], shared_targets["B"], "A vs B"),
            _compare_target_summaries(shared_targets["B"], shared_targets["C"], "B vs C"),
            _compare_target_summaries(shared_targets["A"], shared_targets["C"], "A vs C"),
        ]
        target_c = json.loads(shared_targets["C"].read_text(encoding="utf-8"))
        summary["fixed_targets"] = {
            "identical": all(item.get("identical") for item in target_pairs),
            "comparisons": target_pairs,
            "raw_games_source_revision": "A",
            "raw_games_path": str(shared_raw_games),
            "raw_games_count": args.fixed_games,
            "sample_count": target_c["sample_count"],
            "field_sha256": target_c["field_sha256"],
            "sample_identity_sha256": target_c["sample_identity_sha256"],
            "selfplay_embedded_targets": {
                "identical": all(item.get("identical") for item in selfplay_target_pairs),
                "comparisons": selfplay_target_pairs,
                "interpretation": "These target sets were built from each revision's own fixed self-play output.",
            },
        }
        summary["stages"]["fixed_learner_targets"] = summary["fixed_targets"]["identical"]

        integration = {}
        for label in ("B", "C"):
            worker_output = output / f"integration-{label}"
            _launch(python, script, "worker-integration", [
                "--repo", repos[label], "--expected-revision", EXPECTED_REVISIONS[label],
                "--checkpoint", checkpoint, "--expected-checkpoint-sha", checkpoint_sha,
                "--games-jsonl", fixed["B"]["games_path"], "--output", worker_output,
            ])
            integration[label] = json.loads((worker_output / "integration.json").read_text(encoding="utf-8"))
        central = _compare_integration(Path(output / "integration-B" / "integration.json"),
                                       Path(output / "integration-C" / "integration.json"))
        summary["central_inference"] = central
        summary["stages"]["central_inference_mcts"] = (
            central.get("identical_positions") and central.get("lost_or_misrouted_rows") == 0
            and central.get("strict_policy_wdl_equal") and central.get("root_mcts_equal"))

        training = {}
        shared_dataset = Path(fixed["A"]["dataset_path"])
        for label in ("A", "B", "C"):
            worker_output = output / f"optimizer-{label}"
            _launch(python, script, "worker-training", [
                "--repo", repos[label], "--expected-revision", EXPECTED_REVISIONS[label],
                "--checkpoint", checkpoint, "--expected-checkpoint-sha", checkpoint_sha,
                "--dataset", shared_dataset, "--output", worker_output,
            ])
            training[label] = worker_output / "training.json"
        optimizer_pairs = [
            _compare_training(training["A"], training["B"]),
            _compare_training(training["B"], training["C"]),
            _compare_training(training["A"], training["C"]),
        ]
        optimizer_c = json.loads(training["C"].read_text(encoding="utf-8"))
        summary["optimizer"] = {
            "identical": all(item.get("identical") for item in optimizer_pairs),
            "comparisons": optimizer_pairs,
            "updates": optimizer_c["updates"],
            "model_hash_final": optimizer_c["model_hash_final"],
            "adam_state_sha256": optimizer_c["final_adam_state"]["sha256"],
            "sampled_row_identity_sha256": sha256_json([
                step["sampled_row_ids"] for step in optimizer_c["steps"]]),
        }
        summary["stages"]["bounded_optimizer"] = (
            summary["optimizer"]["identical"] and optimizer_c["updates"] == 32
            and all(item.get("all_rows_resolved", False) for item in
                    [json.loads(path.read_text(encoding="utf-8")) for path in training.values()]))

        pcr_uninterrupted = output / "pcr-uninterrupted"
        pcr_worker_order = output / "pcr-worker-order"
        _launch_selfplay(
            python, script, mode="pcr", repo=repos["C"], expected_revision=EXPECTED_REVISIONS["C"],
            checkpoint=checkpoint, output=pcr_uninterrupted,
            expected_checkpoint_sha=checkpoint_sha, games=args.pcr_games,
            run_id=PCR_RUN_ID, workers=8, active_games_per_worker=4,
            checkpoint_ref=checkpoint_ref)
        _launch_selfplay(
            python, script, mode="pcr", repo=repos["C"], expected_revision=EXPECTED_REVISIONS["C"],
            checkpoint=checkpoint, output=pcr_worker_order,
            expected_checkpoint_sha=checkpoint_sha, games=args.pcr_games,
            run_id=PCR_RUN_ID, workers=16, active_games_per_worker=4,
            checkpoint_ref=checkpoint_ref)
        resume_output = output / "pcr-resume-check"
        _launch(python, script, "worker-resume-check", [
            "--repo", repos["C"], "--expected-revision", EXPECTED_REVISIONS["C"],
            "--checkpoint", checkpoint, "--expected-checkpoint-sha", checkpoint_sha,
            "--checkpoint-ref", checkpoint_ref, "--shard-root", pcr_uninterrupted / "production-shard",
            "--output", resume_output, "--games", args.pcr_games,
        ])
        baseline_games = _load_game_rows(pcr_uninterrupted / "games.jsonl")
        worker_order_games = _load_game_rows(pcr_worker_order / "games.jsonl")
        resume_data = json.loads((resume_output / "resume.json").read_text(encoding="utf-8"))
        worker_order = compare_pcr_decisions_by_position(baseline_games, worker_order_games)
        baseline_target = json.loads((pcr_uninterrupted / "targets.json").read_text(encoding="utf-8"))
        worker_order_target = json.loads((pcr_worker_order / "targets.json").read_text(encoding="utf-8"))
        resume_target = resume_data["targets"]
        uninterrupted_summary = json.loads((pcr_uninterrupted / "summary.json").read_text(encoding="utf-8"))
        worker_order_summary = json.loads((pcr_worker_order / "summary.json").read_text(encoding="utf-8"))
        baseline_invariants = _pcr_run_invariants(baseline_games, baseline_target,
                                                  uninterrupted_summary)
        worker_order_invariants = _pcr_run_invariants(worker_order_games, worker_order_target,
                                                     worker_order_summary)
        resume_raw_matches = (
            set(baseline_games) == {row["game_id"] for row in resume_data["semantic_games"]}
            and {key: baseline_games[key]["semantic_sha256"] for key in baseline_games}
                == {row["game_id"]: row["sha256"] for row in resume_data["semantic_games"]}
        )
        resume_identity = (
            resume_data["request"] == uninterrupted_summary["resume_shard"]["request"]
            and resume_data["committed_shard_loaded_read_only"]
            and resume_data["input_shard_unchanged"]
            and resume_data["all_pcr_positions_represented"]
            and resume_raw_matches
            and resume_target == baseline_target
            and resume_data["semantic_games_sha256"]
                == uninterrupted_summary["semantic_games_sha256"]
        )
        pcr_runs_complete = (
            len(baseline_games) == args.pcr_games
            and len(worker_order_games) == args.pcr_games
            and uninterrupted_summary["games"] == args.pcr_games
            and worker_order_summary["games"] == args.pcr_games
        )
        pcr_game_ids_same = (
            set(baseline_games) == set(worker_order_games)
            == {f"{PCR_RUN_ID}-game-{index:04d}" for index in range(args.pcr_games)}
        )
        worker_order_identity = (
            worker_order["identical_on_shared_positions"] and pcr_game_ids_same
            and pcr_runs_complete)
        pcr = {
            "games": args.pcr_games,
            "runs_complete": pcr_runs_complete,
            "same_game_ids_across_orderings": pcr_game_ids_same,
            "uninterrupted_output": str(pcr_uninterrupted),
            "worker_order_output": str(pcr_worker_order),
            "committed_shard_reload_output": str(resume_output / "resume.json"),
            "invariants_by_execution": {
                "uninterrupted_workers_8": baseline_invariants,
                "worker_order_workers_16": worker_order_invariants,
            },
            "roots": baseline_invariants["roots"],
            "cheap_roots": baseline_invariants["cheap_roots"],
            "full_roots": baseline_invariants["full_roots"],
            "invalid_cap_noise_eligibility_combinations": (
                baseline_invariants["invalid_cap_noise_eligibility_combinations"]
                + worker_order_invariants["invalid_cap_noise_eligibility_combinations"]),
            "cheap_learner_leakage": (baseline_invariants["cheap_learner_leakage"]
                                       + worker_order_invariants["cheap_learner_leakage"]),
            "missing_full_learner_positions": (baseline_invariants["missing_full_learner_positions"]
                                                + worker_order_invariants["missing_full_learner_positions"]),
            "learner_samples": baseline_target["sample_count"],
            "terminal_targets": (baseline_invariants["passed"] and worker_order_invariants["passed"]),
            "worker_order_identity": worker_order_identity,
            "worker_order_comparison": worker_order,
            "committed_shard_load_identity": resume_identity,
            "resume_semantic_matches_uninterrupted": resume_raw_matches,
            "resume_target_matches_uninterrupted": resume_target == baseline_target,
            "mode_decision_sha256": worker_order["left_decision_sha256"],
            "request_identity": uninterrupted_summary["resume_shard"]["request"],
            "uninterrupted_shard_sha256": uninterrupted_summary["resume_shard"]["target_sha256"],
            "resumed_shard_sha256": resume_data["shard_identity"]["shard"]["sha"],
            "all_cheap_games_observed": baseline_invariants["all_cheap_games_observed"],
            "all_cheap_filter_fixture_test": "tests/test_torus9_pcr.py::test_raw_trajectory_retained_and_all_learner_targets_exclude_cheap",
            "production_resume_test": "tests/test_torus9_pcr.py::test_generation_resume_reuses_raw_shard_and_filtered_replay",
            "execution_orderings": [uninterrupted_summary["execution"], worker_order_summary["execution"]],
        }
        pcr["all_cheap_filter_fixture_passed"] = False
        summary["pcr"] = pcr
        summary["stages"]["pcr_invariants"] = (
            baseline_invariants["passed"] and worker_order_invariants["passed"]
            and pcr_runs_complete and pcr_game_ids_same)
        summary["stages"]["pcr_worker_order"] = pcr["worker_order_identity"]
        summary["stages"]["pcr_committed_shard_load"] = pcr["committed_shard_load_identity"]

        summary["stages"]["focused_regressions"] = _run_focused_tests(
            python, repos["C"], output)
        summary["pcr"]["all_cheap_filter_fixture_passed"] = bool(
            summary["stages"]["focused_regressions"]["passed"])
        summary["pcr"]["production_resume_test_passed"] = bool(
            summary["stages"]["focused_regressions"]["passed"])
        summary["stages"]["pcr_resume"] = bool(
            summary["stages"]["pcr_committed_shard_load"]
            and summary["pcr"]["production_resume_test_passed"])
        safety_after = _check_safety_unchanged(before, checkpoint_ref)
        summary["artifact_safety"] = safety_after
        summary["stages"]["artifact_safety"] = safety_after["unchanged"]
        for name, repo in repos.items():
            summary["stages"].setdefault("source_worktrees_clean", True)
            if _source_status(repo, EXPECTED_REVISIONS[name])["clean"] is not True:
                summary["stages"]["source_worktrees_clean"] = False

        expected_stages = (
            "fixed_selfplay", "fixed_repeatability", "fixed_learner_targets", "central_inference_mcts",
            "bounded_optimizer", "pcr_invariants", "pcr_worker_order", "pcr_resume",
            "focused_regressions", "artifact_safety", "source_worktrees_clean",
        )
        summary["verdict"] = "PASS" if all(summary["stages"].get(name) for name in expected_stages) else "FAIL"
        summary["status"] = "COMPLETED"
    except Exception as exc:
        summary["status"] = "FAILED" if any(
            token in str(exc).lower() for token in
            ("diverged", "mismatch", "technical game", "invalid pcr", "target mismatch", "semantic")) else "INCOMPLETE"
        summary["verdict"] = "FAIL" if summary["status"] == "FAILED" else "INCOMPLETE"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["failed_stage"] = "audit orchestration or worker execution"
    finally:
        try:
            if "artifact_safety" not in summary:
                summary["artifact_safety"] = _check_safety_unchanged(before, checkpoint_ref)
        except Exception as safety_exc:
            summary["artifact_safety"] = {"unchanged": False,
                                          "error": f"{type(safety_exc).__name__}: {safety_exc}"}
            summary["verdict"] = "FAIL"
            summary["status"] = "FAILED"
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(_report_markdown(summary), encoding="utf-8")
        print(f"{summary['verdict']}: report saved to {output / 'report.md'}", flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="run the full 64-game Legion/CUDA gate")
    audit.add_argument("--repo-a", type=Path, required=True)
    audit.add_argument("--repo-b", type=Path, required=True)
    audit.add_argument("--repo-c", type=Path, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--checkpoint-ref", type=Path)
    audit.add_argument("--expected-checkpoint-sha", default=EXPECTED_M233_SHA256)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--fixed-games", type=int, default=64)
    audit.add_argument("--pcr-games", type=int, default=16)
    audit.add_argument("--python", default=sys.executable)

    selfplay = sub.add_parser("worker-selfplay", help=argparse.SUPPRESS)
    selfplay.add_argument("--mode", choices=("fixed", "pcr"), required=True)
    selfplay.add_argument("--repo", type=Path, required=True)
    selfplay.add_argument("--expected-revision", required=True)
    selfplay.add_argument("--checkpoint", type=Path, required=True)
    selfplay.add_argument("--expected-checkpoint-sha", required=True)
    selfplay.add_argument("--output", type=Path, required=True)
    selfplay.add_argument("--games", type=int, required=True)
    selfplay.add_argument("--run-id", required=True)
    selfplay.add_argument("--seed", type=int, required=True)
    selfplay.add_argument("--workers", type=int, required=True)
    selfplay.add_argument("--active-games-per-worker", type=int, required=True)
    selfplay.add_argument("--trace-game-id")
    selfplay.add_argument("--trace-ply", type=int, default=1)
    selfplay.add_argument("--device", choices=("cuda",), default="cuda")
    selfplay.add_argument("--save-shard", action="store_true")
    selfplay.add_argument("--checkpoint-ref", type=Path)

    integration = sub.add_parser("worker-integration", help=argparse.SUPPRESS)
    integration.add_argument("--repo", type=Path, required=True)
    integration.add_argument("--expected-revision", required=True)
    integration.add_argument("--checkpoint", type=Path, required=True)
    integration.add_argument("--expected-checkpoint-sha", required=True)
    integration.add_argument("--games-jsonl", type=Path, required=True)
    integration.add_argument("--output", type=Path, required=True)

    target = sub.add_parser("worker-targets", help=argparse.SUPPRESS)
    target.add_argument("--repo", type=Path, required=True)
    target.add_argument("--expected-revision", required=True)
    target.add_argument("--games-jsonl", type=Path, required=True)
    target.add_argument("--output", type=Path, required=True)

    training = sub.add_parser("worker-training", help=argparse.SUPPRESS)
    training.add_argument("--repo", type=Path, required=True)
    training.add_argument("--expected-revision", required=True)
    training.add_argument("--checkpoint", type=Path, required=True)
    training.add_argument("--expected-checkpoint-sha", required=True)
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)

    resume = sub.add_parser("worker-resume-check", help=argparse.SUPPRESS)
    resume.add_argument("--repo", type=Path, required=True)
    resume.add_argument("--expected-revision", required=True)
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--expected-checkpoint-sha", required=True)
    resume.add_argument("--checkpoint-ref", type=Path, required=True)
    resume.add_argument("--shard-root", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--games", type=int, required=True)

    diagnostic = sub.add_parser("worker-diagnostic", help=argparse.SUPPRESS)
    diagnostic.add_argument("--repo", type=Path, required=True)
    diagnostic.add_argument("--expected-revision", required=True)
    diagnostic.add_argument("--checkpoint", type=Path, required=True)
    diagnostic.add_argument("--expected-checkpoint-sha", required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--state-json", type=Path, required=True)
    diagnostic.add_argument("--master-seed", type=int, required=True)
    diagnostic.add_argument("--game-seed", type=int, required=True)
    diagnostic.add_argument("--ply", type=int, required=True)
    diagnostic.add_argument("--search-seed", type=int, required=True)
    diagnostic.add_argument("--left-root-json", type=Path, required=True)
    diagnostic.add_argument("--right-root-json", type=Path, required=True)

    replay = sub.add_parser("worker-replay-evaluations", help=argparse.SUPPRESS)
    replay.add_argument("--repo", type=Path, required=True)
    replay.add_argument("--expected-revision", required=True)
    replay.add_argument("--stream-json", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    geometry = sub.add_parser("batch-geometry", help="sweep exact frozen M233 observation shapes")
    geometry.add_argument("--repo", type=Path, required=True)
    geometry.add_argument("--expected-revision", default=EXPECTED_REVISIONS["A"])
    geometry.add_argument("--checkpoint", type=Path, required=True)
    geometry.add_argument("--expected-checkpoint-sha", default=EXPECTED_M233_SHA256)
    geometry.add_argument("--trace-json", type=Path, required=True)
    geometry.add_argument("--request-index", type=int, required=True)
    geometry.add_argument("--filler-games-jsonl", type=Path, required=True)
    geometry.add_argument("--same-process-repeats", type=int, default=10)
    geometry.add_argument("--fresh-process-repeats", type=int, default=5)
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--python", default=sys.executable)

    geometry_worker = sub.add_parser("worker-batch-geometry", help=argparse.SUPPRESS)
    geometry_worker.add_argument("--repo", type=Path, required=True)
    geometry_worker.add_argument("--expected-revision", required=True)
    geometry_worker.add_argument("--checkpoint", type=Path, required=True)
    geometry_worker.add_argument("--expected-checkpoint-sha", required=True)
    geometry_worker.add_argument("--trace-json", type=Path, required=True)
    geometry_worker.add_argument("--request-index", type=int, required=True)
    geometry_worker.add_argument("--filler-games-jsonl", type=Path, required=True)
    geometry_worker.add_argument("--output", type=Path, required=True)
    geometry_worker.add_argument("--scope", choices=("same-process", "fresh-process"), required=True)
    geometry_worker.add_argument("--repeats", type=int, required=True)
    geometry_worker.add_argument("--repeat-offset", type=int, default=0)

    causal_geometry = sub.add_parser(
        "controlled-root-geometry",
        help="hold root Evaluation fixed and vary batch geometry for leaf Evaluations")
    causal_geometry.add_argument("--repo", type=Path, required=True)
    causal_geometry.add_argument("--expected-revision", default=EXPECTED_REVISIONS["A"])
    causal_geometry.add_argument("--checkpoint", type=Path, required=True)
    causal_geometry.add_argument("--expected-checkpoint-sha", default=EXPECTED_M233_SHA256)
    causal_geometry.add_argument("--trace-json", type=Path, required=True)
    causal_geometry.add_argument("--filler-games-jsonl", type=Path, required=True)
    causal_geometry.add_argument("--repeats", type=int, default=3)
    causal_geometry.add_argument("--output", type=Path, required=True)

    trace_compare = sub.add_parser("compare-root-traces",
                                   help="find the first divergent simulation in two traces")
    trace_compare.add_argument("--left", type=Path, required=True)
    trace_compare.add_argument("--right", type=Path, required=True)
    trace_compare.add_argument("--output", type=Path, required=True)

    replay_group = sub.add_parser("replay-evaluations",
                                  help="replay one exact Evaluation stream on revisions A/B/C")
    replay_group.add_argument("--trace-json", type=Path, required=True)
    replay_group.add_argument("--repo-a", type=Path, required=True)
    replay_group.add_argument("--repo-b", type=Path, required=True)
    replay_group.add_argument("--repo-c", type=Path, required=True)
    replay_group.add_argument("--output", type=Path, required=True)
    replay_group.add_argument("--python", default=sys.executable)

    core_gate = sub.add_parser("deterministic-core",
                               help="run one-worker A/A/B/C complete self-play parity")
    core_gate.add_argument("--repo-a", type=Path, required=True)
    core_gate.add_argument("--repo-b", type=Path, required=True)
    core_gate.add_argument("--repo-c", type=Path, required=True)
    core_gate.add_argument("--checkpoint", type=Path, required=True)
    core_gate.add_argument("--expected-checkpoint-sha", default=EXPECTED_M233_SHA256)
    core_gate.add_argument("--games", type=int, default=16)
    core_gate.add_argument("--output", type=Path, required=True)
    core_gate.add_argument("--python", default=sys.executable)

    core_compare = sub.add_parser(
        "compare-core-runs",
        help="compare completed A1/A2/B/C deterministic-core outputs without rerunning them")
    for label in ("a1", "a2", "b", "c"):
        core_compare.add_argument(f"--{label}", type=Path, required=True)
    core_compare.add_argument("--games", type=int, default=16)
    core_compare.add_argument("--output", type=Path, required=True)

    concurrent_report = sub.add_parser("compare-concurrent",
                                       help="compare five A/A runs with A/B/B/C/C results")
    concurrent_report.add_argument("--a-run", action="append", required=True,
                                   help="revision-A 64-game output directory; pass five times")
    concurrent_report.add_argument("--b-run", type=Path, required=True)
    concurrent_report.add_argument("--c-run", type=Path, required=True)
    concurrent_report.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "audit":
        summary = run_audit(args)
        return 0 if summary.get("verdict") == "PASS" else 1
    if args.command == "worker-selfplay":
        worker_selfplay(args, mode=args.mode)
        return 0
    if args.command == "worker-integration":
        worker_integration(args)
        return 0
    if args.command == "worker-targets":
        worker_targets(args)
        return 0
    if args.command == "worker-training":
        worker_training(args)
        return 0
    if args.command == "worker-resume-check":
        worker_resume_check(args)
        return 0
    if args.command == "worker-diagnostic":
        worker_diagnostic(args)
        return 0
    if args.command == "worker-replay-evaluations":
        worker_replay_evaluations(args)
        return 0
    if args.command == "batch-geometry":
        run_batch_geometry(args)
        return 0
    if args.command == "worker-batch-geometry":
        worker_batch_geometry(args)
        return 0
    if args.command == "controlled-root-geometry":
        report = worker_controlled_geometry_search(args)
        print(json.dumps({"status": report["status"],
                          "report": str(args.output / "controlled-geometry-causality.json")},
                         sort_keys=True), flush=True)
        return 0 if report["status"] == "PROVEN_FOR_THIS_ROOT" else 1
    if args.command == "compare-root-traces":
        compare_root_trace_files(args.left, args.right, args.output)
        return 0
    if args.command == "replay-evaluations":
        run_replay_evaluations(args)
        return 0
    if args.command == "deterministic-core":
        report = run_deterministic_core(args)
        print(json.dumps(report, sort_keys=True, indent=2), flush=True)
        return 0 if report["status"] == "PASS" else 1
    if args.command == "compare-core-runs":
        report = compare_existing_deterministic_core(args)
        print(json.dumps({"status": report["status"],
                          "report": str(args.output / "deterministic-core.json")},
                         sort_keys=True), flush=True)
        return 0 if report["status"] == "PASS" else 1
    if args.command == "compare-concurrent":
        run_concurrent_report(args)
        return 0
    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
