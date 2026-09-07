from alphazero.envs.gocube.parameter_origins import classify_parameters


def test_every_effective_parameter_has_a_recorded_origin_category():
    config = {
        "gocube_katago_reference_commit": "pinned",
        "gocube_size": 4,
        "numMCTSSims": 50,
        "cuda": False,
        "unusual_new_parameter": "still-recorded",
    }
    origins = classify_parameters(config)
    assert set(origins) == set(config)
    assert origins["gocube_katago_reference_commit"] == "katago_reference"
    assert origins["gocube_size"] == "topology_adaptation"
    assert origins["numMCTSSims"] == "experiment"
    assert origins["unusual_new_parameter"] == "framework"
