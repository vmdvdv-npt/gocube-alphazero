from dataclasses import replace

import pytest
import torch

from gocube_golden.rules import prepare_legal_actions
from gocube_golden.scoring import score_terminal
from gocube_golden.state import initial_state, rules_fingerprint_for
from gocube_golden.torus9_contract import TORUS9_POINT_COUNT
from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    Torus9M137FiveChannelGraphNet,
    build_m137_five_channel_observation,
    convert_m137_model,
    save_converted_checkpoint,
)
from gocube_golden.torus9_monolith import (
    Torus9CurrentGraphNet,
    build_torus9_observation,
)
from gocube_golden.topology import TORUS_9X9


def test_m137_affine_conversion_preserves_inference_outputs() -> None:
    torch.manual_seed(137)
    source = Torus9CurrentGraphNet()
    converted = convert_m137_model(source)
    state = initial_state(topology=TORUS_9X9, komi=0.5)
    context = prepare_legal_actions(state)
    legacy = build_torus9_observation(state, legal_context=context).unsqueeze(0)
    five_channel = build_m137_five_channel_observation(state, legal_context=context).unsqueeze(0)
    with torch.inference_mode():
        source_projection = source.input_projection(legacy.transpose(1, 2))
        converted_projection = converted.input_projection(five_channel.transpose(1, 2))
        source_outputs = source.forward_auxiliary(legacy)
        converted_outputs = converted.forward_auxiliary(five_channel)
    assert tuple(five_channel.shape) == (1, 5, TORUS9_POINT_COUNT)
    assert float((source_projection - converted_projection).abs().max()) <= 1e-5
    for source_output, converted_output in zip(source_outputs, converted_outputs):
        assert float((source_output - converted_output).abs().max()) <= 1e-5


def test_m137_observation_ignores_actual_komi_but_referee_does_not() -> None:
    observations = []
    terminal_scores = []
    for komi in (0.5, 2.5, 4.5):
        state = initial_state(topology=TORUS_9X9, komi=komi)
        observations.append(build_m137_five_channel_observation(state))
        terminal = replace(
            state,
            consecutive_passes=2,
            rules_fingerprint=rules_fingerprint_for(TORUS_9X9, komi),
        )
        terminal_scores.append(score_terminal(terminal).margin_black)
    assert all(torch.equal(observations[0], observation) for observation in observations[1:])
    assert terminal_scores == [-0.5, -2.5, -4.5]


def test_m137_derived_checkpoint_is_inference_only(tmp_path) -> None:
    torch.manual_seed(137)
    source = Torus9CurrentGraphNet()
    converted = convert_m137_model(source)
    source_path = tmp_path / "M137.pt"
    source_path.write_bytes(b"canonical-source-placeholder")
    metadata = save_converted_checkpoint(
        tmp_path / "M137-5CH.pt",
        model=converted,
        source_checkpoint=source_path,
        source_metadata={
            "architecture_id": "GoldenGraphNetV2-Torus9",
            "model_hash": "sha256:source",
        },
        converter_git_commit="a" * 40,
    )
    assert metadata["architecture_id"] == M137_FIVE_CHANNEL_ARCHITECTURE_ID
    assert metadata["observation_shape"] == [5, TORUS9_POINT_COUNT]
    assert metadata["optimizer_conversion"] == "not_performed"
    assert metadata["training_ready"] is False
