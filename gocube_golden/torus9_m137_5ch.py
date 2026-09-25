"""Inference-only 5-channel conversion of the canonical Torus9 M137 model.

The legacy six-channel model and loader remain untouched.  This module owns
the explicitly derived architecture and the deterministic 6->5 conversion;
it deliberately does not expose an optimizer conversion or a training path.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch

from .neural import model_hash
from .provenance import file_sha256
from .rules import LegalActionContext, prepare_legal_actions
from .search import Evaluation, SearchError
from .state import BLACK, WHITE, GoldenState
from .torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_POINT_COUNT,
)
from .torus9_monolith import (
    TORUS9_TOPOLOGY_FINGERPRINT,
    Torus9CurrentGraphNet,
    torus9_load_checkpoint,
)


M137_FIVE_CHANNEL_ARCHITECTURE_ID = "GoldenGraphNetV2-Torus9-M137-5CH"
M137_FIVE_CHANNEL_CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_color",
    "previous_pass",
    "legal_point_mask",
)
M137_FIVE_CHANNEL_FORMULA = (
    "W5 = W[:, 0:5]; b5 = b + 0.5 * W[:, 5]; "
    "all remaining parameters copied without modification"
)


class Torus9M137FiveChannelGraphNet(Torus9CurrentGraphNet):
    """The M137-derived network with no komi input channel."""

    architecture_id = M137_FIVE_CHANNEL_ARCHITECTURE_ID

    def __init__(self) -> None:
        super().__init__(hidden=TORUS9_CURRENT_HIDDEN, blocks=TORUS9_CURRENT_BLOCKS)
        self.architecture_id = M137_FIVE_CHANNEL_ARCHITECTURE_ID
        self.input_projection = torch.nn.Linear(5, TORUS9_CURRENT_HIDDEN)

    @property
    def architecture_config(self) -> dict[str, object]:
        config = super().architecture_config
        config["architecture_id"] = M137_FIVE_CHANNEL_ARCHITECTURE_ID
        config["input_channels"] = 5
        config["source_architecture_id"] = TORUS9_CURRENT_ARCHITECTURE_ID
        config["source_checkpoint"] = "M137"
        config["conversion"] = "linear-input-komi-fold-v1"
        config["training_ready"] = False
        return config

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if tuple(observation.shape[1:]) != (5, TORUS9_POINT_COUNT):
            raise ValueError("M137-derived Torus9 network expects [batch,5,81]")
        nodes = self.input_projection(observation.transpose(1, 2))
        for block in self.blocks:
            nodes = block(nodes)
        return torch.nn.functional.relu(self.output_norm(nodes))


def build_m137_five_channel_observation(
    state: GoldenState,
    *,
    legal_context: LegalActionContext | None = None,
) -> torch.Tensor:
    """Build exactly `[5,81]`; referee komi is intentionally not consulted."""

    if state.is_terminal:
        raise ValueError("Terminal Torus9 states must never be observed")
    if state.topology.fingerprint != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("M137-derived observation requires canonical Torus9")
    context = legal_context if legal_context is not None else prepare_legal_actions(state)
    context.assert_compatible(state)
    own = state.side_to_move
    other = WHITE if own == BLACK else BLACK
    observation = torch.zeros((5, TORUS9_POINT_COUNT), dtype=torch.float32)
    for point, stone in enumerate(state.stones):
        observation[0, point] = float(stone == own)
        observation[1, point] = float(stone == other)
    observation[2].fill_(1.0 if own == BLACK else -1.0)
    observation[3].fill_(1.0 if state.consecutive_passes == 1 else 0.0)
    observation[4].copy_(torch.tensor(context.action_mask[:TORUS9_POINT_COUNT], dtype=torch.float32))
    return observation


def load_canonical_m137(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Torus9CurrentGraphNet, dict[str, object]]:
    """Load and validate the six-channel canonical checkpoint."""

    path = Path(checkpoint_path).resolve()
    metadata_path = path.with_suffix(".metadata.json")
    import json

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("M137 metadata must be an object")
    model = Torus9CurrentGraphNet()
    loaded = torus9_load_checkpoint(
        path,
        model=model,
        expected={"model_hash": raw.get("model_hash")},
        device=device,
    )
    return model, dict(loaded)


def convert_m137_model(
    source_model: Torus9CurrentGraphNet,
) -> Torus9M137FiveChannelGraphNet:
    """Apply the exact first-layer affine conversion and copy all other weights."""

    target = Torus9M137FiveChannelGraphNet()
    source_state = source_model.state_dict()
    target_state = target.state_dict()
    source_weight = source_model.input_projection.weight.detach()
    source_bias = source_model.input_projection.bias.detach()
    with torch.no_grad():
        target.input_projection.weight.copy_(source_weight[:, :5])
        target.input_projection.bias.copy_(source_bias + 0.5 * source_weight[:, 5])
    for name, value in source_state.items():
        if name in {"input_projection.weight", "input_projection.bias"}:
            continue
        if name not in target_state:
            raise ValueError(f"M137 conversion target is missing parameter {name}")
        target_state[name].copy_(value)
    target.load_state_dict(target_state, strict=True)
    return target


def save_converted_checkpoint(
    path: str | Path,
    *,
    model: Torus9M137FiveChannelGraphNet,
    source_checkpoint: Path,
    source_metadata: Mapping[str, object],
    converter_git_commit: str,
) -> dict[str, object]:
    """Persist a model-only derived checkpoint; optimizer state is never saved."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_schema_version": 1,
        "checkpoint_label": "M137-5CH-derived",
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "architecture_config": model.architecture_config,
        "observation_shape": [5, TORUS9_POINT_COUNT],
        "source_architecture_id": source_metadata.get("architecture_id"),
        "source_checkpoint": "M137",
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_model_hash": source_metadata.get("model_hash"),
        "converted_model_hash": model_hash(model),
        "converter_git_commit": converter_git_commit,
        "conversion_formula": M137_FIVE_CHANNEL_FORMULA,
        "optimizer_conversion": "not_performed",
        "training_ready": False,
        "parent_lineage": "torus9-m125-continuous-v2-gen6-20260922-v1",
        "parent_checkpoint": "M137",
        "topology_id": "torus-9x9-row-major-v1",
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "heads": {
            "policy": [TORUS9_ACTION_COUNT],
            "wdl": [3],
            "ownership": [TORUS9_POINT_COUNT, 3],
            "score": [1],
        },
    }
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
        },
        destination,
    )
    metadata["artifact_sha256"] = file_sha256(destination)
    destination.with_suffix(".metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


class Torus9M137FiveChannelEvaluator:
    """Inference boundary for the M137-derived model."""

    def __init__(
        self,
        model: Torus9M137FiveChannelGraphNet,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.nn_evaluations = 0

    def evaluate(
        self,
        state: GoldenState,
        *,
        legal_context: LegalActionContext | None = None,
    ) -> Evaluation:
        context = legal_context if legal_context is not None else prepare_legal_actions(state)
        context.assert_compatible(state)
        observation = build_m137_five_channel_observation(state, legal_context=context).to(self.device)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        policy_values = tuple(float(value) for value in policy.detach().cpu())
        wdl_values = tuple(float(value) for value in wdl.detach().cpu())
        if len(policy_values) != TORUS9_ACTION_COUNT or len(wdl_values) != 3:
            raise SearchError("M137-derived Torus 9×9 neural head shape drift")
        if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
            raise SearchError("M137-derived Torus 9×9 neural output is non-finite")
        return Evaluation(policy=policy_values, wdl=wdl_values)


__all__ = [
    "M137_FIVE_CHANNEL_ARCHITECTURE_ID",
    "M137_FIVE_CHANNEL_CHANNELS",
    "M137_FIVE_CHANNEL_FORMULA",
    "Torus9M137FiveChannelGraphNet",
    "Torus9M137FiveChannelEvaluator",
    "build_m137_five_channel_observation",
    "convert_m137_model",
    "load_canonical_m137",
    "save_converted_checkpoint",
]
