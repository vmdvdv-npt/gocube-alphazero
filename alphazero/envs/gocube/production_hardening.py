from __future__ import annotations

import os
from glob import glob

import torch
from torch.utils.data import TensorDataset

from alphazero.envs.gocube.atomic_io import (
    RECOVERY_CONTRACT,
    REPLAY_TENSOR_SUFFIXES,
    cleanup_staging_directory,
    find_last_valid_contiguous_checkpoint,
    load_replay_marker,
    make_staging_directory,
    promote_staged_file,
    remove_replay_marker,
    temporary_sibling,
    write_replay_marker,
)
from alphazero.envs.gocube.exploration_contract import KATAGO_PINNED_EXPLORATION_DEFAULTS
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.katago_train import (
    KataGoSearchCoach,
    assert_fresh_run,
    build_katago_training_args,
    parse_args,
    print_katago_search_configuration,
)
from alphazero.envs.gocube.sample_clock import SampleClockNNetWrapper
from alphazero.envs.gocube.train import validate_tensor_row_counts
from alphazero.utils import get_iter_file


_SEARCH_SELECTION_CONTRACT_FIELDS = (
    "gocube_chosen_move_temperature_early",
    "gocube_chosen_move_temperature",
    "gocube_chosen_move_temperature_halflife",
    "gocube_chosen_move_subtract",
    "gocube_chosen_move_prune",
    "gocube_use_lcb_for_selection",
    "gocube_lcb_stdevs",
    "gocube_min_visit_prop_for_lcb",
    "gocube_value_weight_exponent",
)


class AtomicSampleClockNNetWrapper(SampleClockNNetWrapper):
    """Sample-clock network with a fail-closed atomic recovery contract."""

    def __init__(self, game_cls, args):
        if getattr(args, "gocube_recovery_contract", None) != RECOVERY_CONTRACT:
            raise ValueError(f"Production network requires {RECOVERY_CONTRACT}")
        super().__init__(game_cls, args)

    def _checkpoint_contract(self):
        fields = super()._checkpoint_contract()
        recovery = getattr(self.args, "gocube_recovery_contract", None)
        if recovery != RECOVERY_CONTRACT:
            raise ValueError(f"Missing required recovery contract {RECOVERY_CONTRACT}")
        fields["gocube_recovery_contract"] = recovery
        for key in _SEARCH_SELECTION_CONTRACT_FIELDS:
            if key not in self.args:
                raise ValueError(f"Missing required production search-selection field: {key}")
            fields[key] = self.args[key]
        return fields

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar", make_dirs=True):
        target = os.path.abspath(os.path.join(folder, filename))
        if make_dirs:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        staged = temporary_sibling(target, tag="checkpoint")
        try:
            # Reuse the tested checkpoint payload implementation, but never let
            # it write directly to the visible production filename.
            super().save_checkpoint(folder="", filename=staged, make_dirs=False)
            promote_staged_file(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)


class HardenedKataGoSearchCoach(KataGoSearchCoach):
    """Production coach with atomic replay commits and contiguous resume."""

    def __init__(self, game_cls, nnet, args):
        checkpoint_folder = os.path.join(args.checkpoint, args.run_name)
        existing = glob(os.path.join(checkpoint_folder, "iteration-*.pkl"))
        if args.load_model and existing:
            last_valid, ignored = find_last_valid_contiguous_checkpoint(checkpoint_folder)
            if last_valid is None:
                raise RuntimeError("Existing checkpoint namespace has no resumable checkpoint")

            # Avoid Coach's historical len(glob(...)) resume rule. Initialize
            # its runtime machinery without loading, then load exactly the last
            # structurally valid member of the contiguous 0..N prefix.
            init_args = args.copy()
            init_args.load_model = False
            init_args.startIter = int(last_valid) + 1
            super().__init__(game_cls, nnet, init_args)
            self.args.load_model = True
            self._load_model(self.train_net, int(last_valid))
            # load_checkpoint reinitializes the wrapper, so reconnect Coach's
            # multiprocessing control events afterwards.
            self.train_net.stop_train = self.stop_train
            self.train_net.pause_train = self.pause_train
            self.self_play_iter = int(last_valid)
            self.model_iter = int(last_valid) + 1
            self.args.startIter = int(last_valid) + 1
            if ignored:
                print(
                    "Recovery: ignoring trailing checkpoint iterations after the last valid "
                    f"contiguous checkpoint {last_valid}: {ignored}"
                )
        else:
            super().__init__(game_cls, nnet, args)

    def saveIterationSamples(self, iteration):
        """Commit all six replay tensors as one recoverable logical unit."""

        original_data = self.args.data
        final_folder = os.path.join(original_data, self.args.run_name)
        final_base = os.path.join(
            final_folder,
            get_iter_file(iteration).replace(".pkl", ""),
        )
        os.makedirs(final_folder, exist_ok=True)

        # If this iteration is being replayed after a crash, invalidate the old
        # logical commit before replacing any tensor file.
        remove_replay_marker(final_base)
        staging_root = make_staging_directory(
            original_data,
            prefix=f"{self.args.run_name}-iteration-{int(iteration):04d}",
        )
        try:
            self.args.data = staging_root
            try:
                super().saveIterationSamples(iteration)
            finally:
                self.args.data = original_data

            staged_base = os.path.join(
                staging_root,
                self.args.run_name,
                get_iter_file(iteration).replace(".pkl", ""),
            )
            for suffix in REPLAY_TENSOR_SUFFIXES:
                staged = staged_base + suffix
                if not os.path.exists(staged):
                    raise RuntimeError(f"Replay staging is incomplete: missing {staged}")
                promote_staged_file(staged, final_base + suffix)

            row_count = int(self._iteration_telemetry.get("saved_total", -1))
            if row_count < 0:
                raise RuntimeError("Replay row count was not recorded before commit")
            write_replay_marker(final_base, iteration=iteration, row_count=row_count)
        finally:
            self.args.data = original_data
            cleanup_staging_directory(staging_root)

    def _load_replay_datasets(self, iteration):
        datasets = []
        loaded_samples = {}
        current_history_size = min(
            max(
                self.args.minTrainHistoryWindow,
                (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters,
            ),
            self.args.maxTrainHistoryWindow,
        )
        for train_iter in range(max(1, iteration - current_history_size), iteration + 1):
            base = os.path.join(
                self.args.data,
                self.args.run_name,
                get_iter_file(train_iter).replace(".pkl", ""),
            )
            try:
                marker = load_replay_marker(base)
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"Warning: ignoring incomplete replay iteration {train_iter}: {exc}")
                continue

            try:
                tensors = [torch.load(base + suffix) for suffix in REPLAY_TENSOR_SUFFIXES]
            except (FileNotFoundError, OSError, RuntimeError, EOFError) as exc:
                print(f"Warning: ignoring unreadable replay iteration {train_iter}: {exc}")
                continue
            row_count = validate_tensor_row_counts(tensors, expected=int(marker["row_count"]))
            if tensors[0].shape[1:] != self.game_cls.observation_size():
                raise ValueError("V3 dataset observation schema/shape mismatch")
            datasets.append(TensorDataset(*tensors))
            loaded_samples[train_iter] = row_count
        return datasets, loaded_samples


def build_hardened_training_args(cli):
    game_cls, args = build_katago_training_args(cli)
    args = args.copy()
    defaults = KATAGO_PINNED_EXPLORATION_DEFAULTS
    # build_training_args() fingerprints the base V3 game before
    # build_katago_training_args() wraps it with pinned passWouldEndPhase /
    # diversification semantics. Checkpoints must describe the game class
    # actually used by the production network, otherwise the first save/load
    # fails closed on a rules-contract mismatch.
    args.gocube_rules_fingerprint = game_cls.rules_fingerprint()
    args.gocube_recovery_contract = RECOVERY_CONTRACT
    args.gocube_chosen_move_temperature_early = defaults["chosen_move_temperature_early"]
    args.gocube_chosen_move_temperature = defaults["chosen_move_temperature"]
    args.gocube_chosen_move_temperature_halflife = defaults["chosen_move_temperature_halflife"]
    args.gocube_chosen_move_subtract = defaults["chosen_move_subtract"]
    args.gocube_chosen_move_prune = defaults["chosen_move_prune"]
    args.gocube_use_lcb_for_selection = defaults["use_lcb_for_selection"]
    args.gocube_lcb_stdevs = defaults["lcb_stdevs"]
    args.gocube_min_visit_prop_for_lcb = defaults["min_visit_prop_for_lcb"]
    args.gocube_value_weight_exponent = defaults["value_weight_exponent"]
    return game_cls, args


def print_hardened_configuration(args):
    print_katago_search_configuration(args)
    print("Pinned KataGo move/value selection:")
    print(
        "  chosen move temp early/late = "
        f"{args.gocube_chosen_move_temperature_early:g}/{args.gocube_chosen_move_temperature:g}"
    )
    print(f"  chosen move halflife = {args.gocube_chosen_move_temperature_halflife:g}")
    print(f"  value weight exponent = {args.gocube_value_weight_exponent:g}")
    print(
        "  LCB = "
        f"{'ON' if args.gocube_use_lcb_for_selection else 'OFF'} "
        f"(stdevs={args.gocube_lcb_stdevs:g}, minVisitProp={args.gocube_min_visit_prop_for_lcb:g})"
    )
    print("Crash recovery:")
    print(f"  contract = {args.gocube_recovery_contract}")
    print("  checkpoint writes = staging + fsync + atomic replace")
    print("  replay commit = 6 tensors + completion marker")
    print("  resume = last valid contiguous checkpoint")


def main(argv=None):
    cli = parse_args(argv)
    game_cls, args = build_hardened_training_args(cli)
    if not cli.allow_existing_run:
        assert_fresh_run(args)
    print_hardened_configuration(args)
    ensure_training_manifest(args.checkpoint, args.run_name, game_cls)
    network = AtomicSampleClockNNetWrapper(game_cls, args)
    coach = HardenedKataGoSearchCoach(game_cls, network, args)
    coach.learn()


if __name__ == "__main__":
    main()
