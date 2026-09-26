"""Production hardening for the M137 Torus9 komi calibration workflow.

The generic calibration implementation deliberately contains the scientific
state machine.  This wrapper adds two production-only guarantees:

* a requested parent soft-stop must be acknowledged before any calibration
  Arena consumes the GPU;
* when the optional second 1024-game batch is required, the winner is selected
  from the cumulative 2048 valid games rather than from the extension batch
  alone.

Batch evidence is persisted separately so a restart between an Arena commit
and cumulative-stat publication cannot lose the first 1024 games.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
from pathlib import Path
import time

from .policy import (
    aggregate_candidate_batches as _aggregate_candidate_batches,
    select_komi,
)
from ...orchestrator_v2.artifact_resolver import ResolvedCheckpointNode
from .runner import (
    CALIBRATION_EXTENSION,
    KOMI_CALIBRATION_CANDIDATES,
    KomiCalibrationError,
    KomiCalibrationRunnerV2 as _BaseKomiCalibrationRunnerV2,
)


_PARENT_STOP_TIMEOUT_SECONDS = 6 * 60 * 60


def aggregate_candidate_batches(*args: object, **kwargs: object) -> dict[str, object]:
    """Compatibility adapter preserving the historical exception type."""
    try:
        return _aggregate_candidate_batches(*args, **kwargs)  # type: ignore[arg-type]
    except ValueError as exc:
        raise KomiCalibrationError(str(exc)) from exc


class ProductionKomiCalibrationRunnerV2(_BaseKomiCalibrationRunnerV2):
    """Production-safe calibration runner used by the V2 CLI entrypoint."""

    def _batch_ledger(
        self, state: MutableMapping[str, object]
    ) -> MutableMapping[str, object]:
        configured = tuple(
            getattr(self.config, "candidates", KOMI_CALIBRATION_CANDIDATES)
        )
        raw = state.get("candidate_batches")
        if raw is None:
            raw = {f"{komi:g}": {} for komi in configured}
            state["candidate_batches"] = raw
        if not isinstance(raw, MutableMapping):
            self._fail(state, "komi calibration batch ledger is malformed")
        for komi in configured:
            key = f"{komi:g}"
            bucket = raw.get(key)
            if bucket is None:
                raw[key] = {}
            elif not isinstance(bucket, MutableMapping):
                self._fail(state, f"komi {key} batch ledger is malformed")
        return raw

    def _capture_current_candidate(
        self, state: MutableMapping[str, object], key: str
    ) -> None:
        candidates = state.get("candidates")
        if not isinstance(candidates, MutableMapping):
            self._fail(state, "calibration candidate state is malformed")
        current = candidates.get(key)
        if not isinstance(current, Mapping) or current.get("aggregate") is True:
            return
        batch = int(current.get("batch", 0))
        if batch <= 0:
            return
        ledger = self._batch_ledger(state)
        bucket = ledger[key]
        assert isinstance(bucket, MutableMapping)
        bucket.setdefault(str(batch), deepcopy(dict(current)))

    def _candidate_batch_evidence(
        self, state: MutableMapping[str, object], key: str
    ) -> list[Mapping[str, object]]:
        self._capture_current_candidate(state, key)
        ledger = self._batch_ledger(state)
        bucket = ledger[key]
        assert isinstance(bucket, Mapping)
        evidence: list[Mapping[str, object]] = []
        for batch in sorted(int(value) for value in bucket):
            raw = bucket.get(str(batch))
            if not isinstance(raw, Mapping):
                self._fail(state, f"komi {key} batch evidence is malformed")
            evidence.append(raw)
        return evidence

    def _publish_aggregate(
        self, state: MutableMapping[str, object], key: str
    ) -> dict[str, object]:
        evidence = self._candidate_batch_evidence(state, key)
        aggregate = aggregate_candidate_batches(evidence)
        candidates = state.get("candidates")
        assert isinstance(candidates, MutableMapping)
        candidates[key] = aggregate
        self._persist(state)
        return aggregate

    def _await_parent_soft_stop(
        self,
        state: MutableMapping[str, object],
        parent: ResolvedCheckpointNode,
    ) -> None:
        if state.get("parent_stop_acknowledged") is True:
            return

        control = parent.owner_root / "control" / "soft-stop.json"
        if not control.is_file():
            # Synthetic/unit runners and already-stopped historical parents do
            # not necessarily expose a continuous-training soft-stop control.
            return

        runtime_state = parent.owner_root / "runtime" / "state.json"
        deadline = time.monotonic() + _PARENT_STOP_TIMEOUT_SECONDS
        poll = max(0.1, float(self.config.wait_poll_seconds))

        while True:
            if runtime_state.is_file():
                try:
                    payload = json.loads(runtime_state.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, Mapping):
                    lifecycle = str(payload.get("state", ""))
                    active_generation = payload.get("active_generation")
                    if lifecycle in {"SOFT_STOPPED", "COMPLETED"} and active_generation is None:
                        state["parent_stop_acknowledged"] = True
                        state["parent_stop_state"] = lifecycle
                        self._persist(state)
                        return
                    if lifecycle in {"FAILED", "CORRUPT"}:
                        self._fail(
                            state,
                            f"parent training entered terminal state {lifecycle} before calibration",
                        )

            if time.monotonic() >= deadline:
                self._fail(
                    state,
                    "parent graceful stop was not acknowledged before calibration timeout",
                )
            self.sleeper(poll)

    def _run_candidate(
        self,
        state: dict[str, object],
        parent: ResolvedCheckpointNode,
        komi: float,
        stage: str,
        *,
        batch: int,
    ) -> None:
        self._await_parent_soft_stop(state, parent)
        key = f"{komi:g}"
        self._capture_current_candidate(state, key)
        ledger = self._batch_ledger(state)
        bucket = ledger[key]
        assert isinstance(bucket, MutableMapping)

        if str(batch) in bucket:
            self._publish_aggregate(state, key)
            return
        if batch > 1 and "1" not in bucket:
            self._fail(state, f"komi {key} extension lacks its first 1024-game batch")

        # Persist the first-batch evidence outside candidates before the base
        # runner temporarily replaces candidates[key] with the extension batch.
        self._persist(state)
        super()._run_candidate(state, parent, komi, stage, batch=batch)
        self._capture_current_candidate(state, key)
        self._publish_aggregate(state, key)

    def _select_or_extend(
        self, state: dict[str, object], parent: ResolvedCheckpointNode
    ) -> float:
        configured = tuple(
            getattr(self.config, "candidates", KOMI_CALIBRATION_CANDIDATES)
        )
        if configured != KOMI_CALIBRATION_CANDIDATES:
            candidates = state.get("candidates")
            if not isinstance(candidates, Mapping):
                self._fail(state, "calibration candidates are malformed")
            biases: dict[float, float] = {}
            for komi in configured:
                key = f"{komi:g}"
                evidence = self._candidate_batch_evidence(state, key)
                first = next(
                    (item for item in evidence if int(item.get("batch", 0)) == 1),
                    None,
                )
                if first is None:
                    self._fail(state, f"komi {key} first calibration batch is missing")
                stats = first.get("stats")
                if not isinstance(stats, Mapping):
                    self._fail(state, f"komi {key} first-batch stats are malformed")
                biases[komi] = float(stats["bias"])
            return select_komi(biases, candidates=configured).selected_komi

        initial_biases: dict[float, float] = {}

        for komi in KOMI_CALIBRATION_CANDIDATES:
            key = f"{komi:g}"
            evidence = self._candidate_batch_evidence(state, key)
            by_batch = {int(item["batch"]): item for item in evidence}
            first = by_batch.get(1)
            if first is None:
                self._fail(state, f"komi {key} first calibration batch is missing")
            stats = first.get("stats")
            if not isinstance(stats, Mapping):
                self._fail(state, f"komi {key} first-batch stats are malformed")
            initial_biases[komi] = float(stats["bias"])

        initial_decision = select_komi(
            initial_biases,
            candidates=KOMI_CALIBRATION_CANDIDATES,
            ambiguity_threshold=self.config.ambiguity_threshold,
        )
        if initial_decision.requires_extension:
            self._transition(state, CALIBRATION_EXTENSION)
            self._run_candidate(state, parent, 1.5, CALIBRATION_EXTENSION, batch=2)
            self._run_candidate(state, parent, 2.5, CALIBRATION_EXTENSION, batch=2)
            cumulative_biases: dict[float, float] = {}
            for komi in KOMI_CALIBRATION_CANDIDATES:
                key = f"{komi:g}"
                aggregate = self._publish_aggregate(state, key)
                stats = aggregate["stats"]
                assert isinstance(stats, Mapping)
                if int(stats["valid_games"]) != (
                    int(self.config.initial_games) + int(self.config.extension_games)
                ):
                    self._fail(
                        state,
                        f"komi {key} extension did not produce 2048 cumulative valid games",
                    )
                cumulative_biases[komi] = float(stats["bias"])
            return select_komi(
                initial_biases,
                candidates=KOMI_CALIBRATION_CANDIDATES,
                ambiguity_threshold=self.config.ambiguity_threshold,
                cumulative_biases=cumulative_biases,
            ).selected_komi

        # If no extension is scientifically required, decide from the first
        # 1024-game batch exactly as specified.  A stray persisted extension is
        # ignored rather than changing the decision rule.
        return initial_decision.selected_komi


__all__ = [
    "ProductionKomiCalibrationRunnerV2",
    "aggregate_candidate_batches",
]
