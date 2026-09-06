from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args


def test_hardened_builder_fingerprints_actual_diversified_game_class():
    game_cls, args = build_hardened_training_args(parse_args([]))

    assert args.gocube_rules_fingerprint == game_cls.rules_fingerprint()
