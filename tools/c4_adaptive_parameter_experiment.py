#!/usr/bin/env python3
"""Compatibility entrypoint for the complete GoCube adaptive sweep."""

from tools.c4_overnight_complete import *  # noqa: F401,F403
from tools.c4_overnight_experiment import Experiment as _CompleteExperiment
from tools import c4_overnight_complete as _impl


class Experiment(_CompleteExperiment):
    """Backward-compatible facade used by existing tests and commands."""

    def training_command(self, *args, **kwargs):
        if not hasattr(self, "selfplay_wait_ms"):
            self.selfplay_wait_ms = _impl.SELFPLAY_BATCH_WAIT_MS
        return super().training_command(*args, **kwargs)


def main(argv=None) -> int:
    Experiment(_impl.parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
