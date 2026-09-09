"""Retired GoCube training entrypoint.

GoCube V3 training is served by the pinned KataGo path.  The compatibility
imports below keep old helper/test imports readable, while the executable
entrypoint fails closed so it cannot start legacy self-play.
"""

from alphazero.envs.gocube.training_common import (
    GoCubeCoach,
    build_base_training_args,
    expected_saved_samples,
    parse_args,
    print_training_configuration,
    resolve_train_steps,
    validate_tensor_row_counts,
    validate_v3_target_tensors,
)


def build_training_args(cli):
    """Compatibility alias for old callers; this does not enable legacy search."""

    return build_base_training_args(cli)


def main():
    raise RuntimeError(
        "GoCube V3 training through this legacy entrypoint is retired. "
        "Use alphazero.envs.gocube.katago_train or "
        "alphazero.envs.gocube.hardened_train."
    )


if __name__ == "__main__":
    main()
