"""Process-isolated execution for the Golden Torus Arena.

This module is deliberately an orchestration layer.  A worker never implements
game rules or search: it constructs its process-local players and delegates one
task to :class:`SequentialGoldenArena.play_game`.  The sequential Arena is
therefore still the reference oracle and the only game execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Union

from .arena import (
    ArenaSummary,
    GameRecord,
    SequentialGoldenArena,
    pair_schedule_from_records,
    recompute_summary,
    validate_game_record,
    validate_pair_records,
)
from .arena_contract import (
    DEFAULT_ARENA_CONTRACT,
    SEARCH_IMPLEMENTATION_ID,
    GoldenArenaContract,
    reject_checkpoint_arena_overrides,
)
from .experiment_profile import (
    EXPERIMENT_FINGERPRINT,
    PROFILE_ID,
    RULES_PROFILE_ID,
    SEED_DERIVATION_ID,
)
from .neural import GoldenNeuralEvaluator, instantiate_model_from_metadata, model_hash
from .players import Player, SearchPlayer
from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    CodeIdentity,
    PlayerIdentity,
    RunManifest,
    capture_code_identity,
    file_sha256,
    infer_player_identity,
    POLICY_SEMANTICS,
    sha256_fingerprint,
)
from .search import SequentialPUCT
from .state import GoldenState, initial_state
from .training import load_checkpoint


_futures = importlib.import_module("concurrent.futures")
Future = _futures.Future
ProcessPoolExecutor = _futures.ProcessPoolExecutor
as_completed = _futures.as_completed
get_context = importlib.import_module("multiprocessing").get_context


@dataclass(frozen=True)
class PlayerSpec:
    """A pickleable description of a non-checkpoint player.

    ``player`` is kept as a process initializer payload, so a stateful player
    object is serialized once per worker rather than once per game.  A factory
    is useful for callers that need a fresh process-local object; it must be a
    pickleable callable (normally a module-level function or callable class).
    The identity is explicit for factories because the parent must be able to
    construct every task without executing a game or guessing provenance.
    """

    player: Player | None = None
    factory: Callable[[], Player] | None = None
    identity: PlayerIdentity | None = None

    def __post_init__(self) -> None:
        if (self.player is None) == (self.factory is None):
            raise ValueError("PlayerSpec requires exactly one player or factory")
        if self.identity is None:
            if self.player is None:
                raise ValueError("PlayerSpec.factory requires an explicit identity")
            identity = infer_player_identity(self.player)
            object.__setattr__(self, "identity", identity)
        self.identity.validate()  # type: ignore[union-attr]

    @classmethod
    def from_player(cls, player: Player) -> "PlayerSpec":
        return cls(player=player)

    @classmethod
    def from_factory(
        cls,
        factory: Callable[[], Player],
        *,
        identity: PlayerIdentity,
    ) -> "PlayerSpec":
        return cls(factory=factory, identity=identity)

    def build(self) -> Player:
        player = self.player if self.factory is None else self.factory()
        if player is None:
            raise RuntimeError("PlayerSpec did not produce a player")
        actual = infer_player_identity(player)
        if actual != self.identity:
            raise ValueError(
                "PlayerSpec factory produced a player with a different structured identity"
            )
        return player


@dataclass(frozen=True)
class CheckpointPlayerSpec:
    """Process-local loader for one immutable Golden checkpoint.

    The checkpoint is loaded in the worker initializer, exactly once per
    worker.  No model object is placed in a ``GameTask`` and no model is
    shared between worker processes.
    """

    checkpoint_path: str
    player_id: str
    metadata_json: str
    artifact_sha256: str
    expected_model_hash: str
    device: str = "cpu"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        player_id: str,
        device: str = "cpu",
        metadata: Mapping[str, object] | None = None,
        artifact_sha256: str | None = None,
    ) -> "CheckpointPlayerSpec":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise ValueError(f"Golden checkpoint artifact does not exist: {path}")
        if metadata is None:
            metadata_path = path.with_suffix(".metadata.json")
            if not metadata_path.is_file():
                raise ValueError(f"Checkpoint metadata sidecar is missing: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        normalized = dict(metadata)
        actual_artifact = file_sha256(path)
        artifact = artifact_sha256 or actual_artifact
        if artifact != actual_artifact:
            raise ValueError("Checkpoint artifact hash does not match the checkpoint file")
        expected_model_hash = str(normalized.get("model_hash", ""))
        if not expected_model_hash:
            raise ValueError("Checkpoint metadata is missing model_hash")
        # Validate all checkpoint provenance before any process is launched.
        _checkpoint_identity(
            player_id=player_id,
            metadata=normalized,
            artifact_sha256=artifact,
            expected_model_hash=expected_model_hash,
        )
        return cls(
            checkpoint_path=str(path),
            player_id=str(player_id),
            metadata_json=json.dumps(normalized, sort_keys=True, separators=(",", ":")),
            artifact_sha256=str(artifact),
            expected_model_hash=expected_model_hash,
            device=str(device),
        )

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self.metadata_json)

    @property
    def identity(self) -> PlayerIdentity:
        metadata = self.metadata
        return _checkpoint_identity(
            player_id=self.player_id,
            metadata=metadata,
            artifact_sha256=self.artifact_sha256,
            expected_model_hash=self.expected_model_hash,
        )

    def build(self) -> Player:
        torch = importlib.import_module("torch")
        device = torch.device(self.device)
        model = instantiate_model_from_metadata(self.metadata).to(device)
        loaded = load_checkpoint(
            self.checkpoint_path,
            model=model,
            expected={"model_hash": self.expected_model_hash},
            device=device,
        )
        if model_hash(model) != self.expected_model_hash:
            raise RuntimeError(
                f"Worker loaded checkpoint with wrong model hash: {self.checkpoint_path}"
            )
        if str(loaded.get("artifact_sha256")) != self.artifact_sha256:
            raise RuntimeError(
                f"Worker loaded checkpoint with wrong artifact hash: {self.checkpoint_path}"
            )
        evaluator = GoldenNeuralEvaluator(model, device=device)
        # SearchPlayer's structured identity is derived from these fields.
        evaluator.checkpoint_path = self.checkpoint_path
        evaluator.checkpoint_metadata = self.metadata
        return SearchPlayer(
            self.player_id,
            SequentialPUCT(),
            evaluator,
            identity=self.identity,
        )


PlayerSpecLike = Union[PlayerSpec, CheckpointPlayerSpec]


def _checkpoint_identity(
    *,
    player_id: str,
    metadata: Mapping[str, object],
    artifact_sha256: str,
    expected_model_hash: str,
) -> PlayerIdentity:
    """Build the same identity used by the existing Stage-4 Arena runner.

    ``model_hash`` identifies model parameters while ``artifact_sha256``
    identifies the serialized checkpoint file; they are intentionally distinct
    fields in the existing checkpoint contract.
    """

    reject_checkpoint_arena_overrides(metadata)
    required = (
        "observation_schema_id",
        "observation_fingerprint",
        "target_contract_id",
        "target_fingerprint",
        "value_head_semantics",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError("Checkpoint metadata is incomplete: " + ", ".join(missing))
    identity = PlayerIdentity(
        logical_player_id=str(player_id),
        player_kind="checkpoint",
        source_identity=f"stage4-checkpoint:{player_id}:{expected_model_hash}",
        model_file_sha256=str(artifact_sha256),
        checkpoint_metadata_fingerprint=sha256_fingerprint(dict(metadata)),
        observation_contract_id=str(metadata["observation_schema_id"]),
        observation_fingerprint=str(metadata["observation_fingerprint"]),
        target_contract_id=str(metadata["target_contract_id"]),
        target_fingerprint=str(metadata["target_fingerprint"]),
        value_semantics=str(metadata["value_head_semantics"]),
        policy_semantics=POLICY_SEMANTICS,
    )
    identity.validate()
    return identity


def _coerce_player_spec(player: Player | PlayerSpecLike) -> PlayerSpecLike:
    if isinstance(player, (PlayerSpec, CheckpointPlayerSpec)):
        return player
    return PlayerSpec.from_player(player)


@dataclass(frozen=True)
class PairTask:
    """One deterministic Arena start; it expands to A-black and B-black games."""

    pair_id: str
    start_state: GoldenState | None = None
    start_trace: tuple[int | str, ...] = ()
    game_ids: tuple[str, str] | None = None
    allow_research_komi: bool = False


ArenaPairTask = PairTask


@dataclass(frozen=True)
class GameTask:
    """Complete immutable input for exactly one Arena game.

    ``ordinal`` is the parent-declared canonical schedule position.  It is
    never derived from completion order and is not used for seed derivation.
    """

    ordinal: int
    pair_id: str
    game_id: str
    black_player: str
    start_state: GoldenState
    start_trace: tuple[int | str, ...]
    master_seed: int
    run_id: str
    contract: GoldenArenaContract
    code_identity: CodeIdentity
    player_A_identity: PlayerIdentity
    player_B_identity: PlayerIdentity
    allow_research_komi: bool = False

    @property
    def checkpoint_A_identity(self) -> PlayerIdentity:
        """Compatibility name for the task's immutable A-slot identity."""

        return self.player_A_identity

    @property
    def checkpoint_B_identity(self) -> PlayerIdentity:
        """Compatibility name for the task's immutable B-slot identity."""

        return self.player_B_identity


class ArenaWorkerError(RuntimeError):
    """A worker failure that invalidates the whole Arena collection."""

    def __init__(self, message: str, *, task: GameTask | None = None) -> None:
        super().__init__(message)
        self.task = task


_WORKER_PLAYER_A: Player | None = None
_WORKER_PLAYER_B: Player | None = None
_WORKER_IDENTITIES: tuple[PlayerIdentity, PlayerIdentity] | None = None


def _worker_initializer(
    player_A_spec: PlayerSpecLike,
    player_B_spec: PlayerSpecLike,
    player_A_identity: PlayerIdentity,
    player_B_identity: PlayerIdentity,
) -> None:
    global _WORKER_PLAYER_A, _WORKER_PLAYER_B, _WORKER_IDENTITIES
    # Match the existing Torus Arena caller's fixed CPU execution contract in
    # each spawned process.  This avoids oversubscription; search itself is
    # unchanged and still executes sequentially inside each game.
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    # The builders are called once in each persistent worker process.  In
    # particular, CheckpointPlayerSpec loads each checkpoint once per worker.
    player_A = player_A_spec.build()
    player_B = player_B_spec.build()
    if infer_player_identity(player_A) != player_A_identity:
        raise ValueError("Worker A player identity does not match GameTask identity")
    if infer_player_identity(player_B) != player_B_identity:
        raise ValueError("Worker B player identity does not match GameTask identity")
    _WORKER_PLAYER_A = player_A
    _WORKER_PLAYER_B = player_B
    _WORKER_IDENTITIES = (player_A_identity, player_B_identity)


def _validate_worker_record(record: GameRecord, task: GameTask) -> None:
    validate_game_record(record)
    checks = {
        "pair_id": (record.pair_id, task.pair_id),
        "game_id": (record.game_id, task.game_id),
        "black_player": (record.black_player, task.black_player),
        "master_seed": (record.master_seed, task.master_seed),
        "run_id": (record.run_id, task.run_id),
        "start_state_key": (record.start_state_key, task.start_state.state_key),
        "start_history": (record.start_history, task.start_state.superko_history),
        "start_trace": (record.start_trace, task.start_trace),
        "player_A_identity_fingerprint": (
            record.player_A_identity_fingerprint,
            task.player_A_identity.fingerprint,
        ),
        "player_B_identity_fingerprint": (
            record.player_B_identity_fingerprint,
            task.player_B_identity.fingerprint,
        ),
        "git_commit_sha": (record.git_commit_sha, task.code_identity.git_commit_sha),
        "git_tree_sha": (record.git_tree_sha, task.code_identity.git_tree_sha),
        "git_worktree_clean": (record.git_worktree_clean, task.code_identity.working_tree_clean),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Worker GameRecord {name} disagrees with GameTask")
    if record.search_settings != task.contract.search.evidence():
        raise ValueError("Worker GameRecord search contract disagrees with GameTask")


def _worker_play_game(task: GameTask) -> GameRecord:
    if _WORKER_PLAYER_A is None or _WORKER_PLAYER_B is None or _WORKER_IDENTITIES is None:
        raise RuntimeError("Arena worker was not initialized")
    arena = SequentialGoldenArena(
        contract=task.contract,
        master_seed=task.master_seed,
        run_id=task.run_id,
        code_identity=task.code_identity,
    )
    record = arena.play_game(
        game_id=task.game_id,
        pair_id=task.pair_id,
        player_A=_WORKER_PLAYER_A,
        player_B=_WORKER_PLAYER_B,
        black_player=task.black_player,
        start_state=task.start_state,
        start_trace=task.start_trace,
        allow_research_komi=task.allow_research_komi,
    )
    _validate_worker_record(record, task)
    return record


class ProcessParallelGoldenArena:
    """Execute independent Golden games in persistent OS worker processes."""

    MAX_WORKERS = 16

    def __init__(
        self,
        *,
        player_A: Player | PlayerSpecLike,
        player_B: Player | PlayerSpecLike,
        workers: int = 16,
        contract: GoldenArenaContract = DEFAULT_ARENA_CONTRACT,
        master_seed: int = 0,
        run_id: str | None = None,
        code_identity: CodeIdentity | None = None,
        require_canonical_code: bool = False,
        mp_context: str | None = None,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= self.MAX_WORKERS:
            raise ValueError(f"Golden Arena workers must be an integer in 1..{self.MAX_WORKERS}")
        self.contract = contract
        self.master_seed = int(master_seed)
        self.run_id = run_id or "golden-process-parallel"
        self.code_identity = code_identity or capture_code_identity()
        self.code_identity.validate(require_canonical=require_canonical_code)
        self.player_A_spec = _coerce_player_spec(player_A)
        self.player_B_spec = _coerce_player_spec(player_B)
        self.player_A_identity = self.player_A_spec.identity
        self.player_B_identity = self.player_B_spec.identity
        if self.player_A_identity is None or self.player_B_identity is None:
            raise ValueError("Process Arena player specs require structured identities")
        self.player_A_identity.validate()
        self.player_B_identity.validate()
        self.workers = workers
        self.mp_context = mp_context
        self._records: list[GameRecord] = []
        self._game_ids: set[str] = set()
        self._pair_ids: set[str] = set()
        self._failed = False
        self._last_schedule: tuple[tuple[str, str, str], ...] = ()

    @property
    def records(self) -> tuple[GameRecord, ...]:
        return tuple(self._records)

    @property
    def canonical_schedule(self) -> tuple[tuple[str, str, str], ...]:
        return self._last_schedule

    def _make_tasks(self, pairs: Sequence[PairTask]) -> tuple[GameTask, ...]:
        tasks: list[GameTask] = []
        seen_game_ids: set[str] = set(self._game_ids)
        seen_pair_ids: set[str] = set(self._pair_ids)
        for pair_index, pair in enumerate(pairs):
            if not pair.pair_id or pair.pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate or empty pair_id rejected: {pair.pair_id!r}")
            seen_pair_ids.add(pair.pair_id)
            start_state = pair.start_state or initial_state()
            first_id, second_id = pair.game_ids or (
                f"{pair.pair_id}-g1",
                f"{pair.pair_id}-g2",
            )
            if not first_id or not second_id or first_id == second_id:
                raise ValueError("Duplicate game_id or empty game_id in Arena pair")
            if first_id in seen_game_ids or second_id in seen_game_ids:
                raise ValueError("Duplicate game_id rejected before worker launch")
            seen_game_ids.update((first_id, second_id))
            base = pair_index * 2
            tasks.extend(
                (
                    GameTask(
                        ordinal=base,
                        pair_id=pair.pair_id,
                        game_id=first_id,
                        black_player="A",
                        start_state=start_state,
                        start_trace=tuple(pair.start_trace),
                        master_seed=self.master_seed,
                        run_id=self.run_id,
                        contract=self.contract,
                        code_identity=self.code_identity,
                        player_A_identity=self.player_A_identity,
                        player_B_identity=self.player_B_identity,
                        allow_research_komi=pair.allow_research_komi,
                    ),
                    GameTask(
                        ordinal=base + 1,
                        pair_id=pair.pair_id,
                        game_id=second_id,
                        black_player="B",
                        start_state=start_state,
                        start_trace=tuple(pair.start_trace),
                        master_seed=self.master_seed,
                        run_id=self.run_id,
                        contract=self.contract,
                        code_identity=self.code_identity,
                        player_A_identity=self.player_A_identity,
                        player_B_identity=self.player_B_identity,
                        allow_research_komi=pair.allow_research_komi,
                    ),
                )
            )
        return tuple(tasks)

    def build_tasks(self, pairs: Iterable[PairTask]) -> tuple[GameTask, ...]:
        """Materialize and validate the entire canonical schedule in the parent."""

        raw = tuple(pairs)
        return self._make_tasks(raw)

    def _context(self):
        if self.mp_context is not None:
            return get_context(self.mp_context)
        # Spawn is the safe default: the parent never shares imported/model
        # state with workers.  Callers may explicitly select fork on a trusted
        # CPU-only deployment for lower process startup overhead.
        return get_context("spawn")

    def play_games(self, tasks: Iterable[GameTask]) -> tuple[GameRecord, ...]:
        """Run a complete parent-declared GameTask list transactionally.

        Futures are observed in completion order only for failure detection.
        The returned tuple is reconstructed by the task list's canonical
        order, so completion order cannot become scientific evidence order.
        """

        if self._failed:
            raise ArenaWorkerError("Process Arena is failed closed after a worker error")
        raw_tasks = tuple(tasks)
        if not raw_tasks:
            return ()
        self._validate_tasks(raw_tasks)
        by_game_id: dict[str, GameRecord] = {}
        futures: dict[Future[GameRecord], GameTask] = {}
        try:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=self._context(),
                initializer=_worker_initializer,
                initargs=(
                    self.player_A_spec,
                    self.player_B_spec,
                    self.player_A_identity,
                    self.player_B_identity,
                ),
            ) as pool:
                for task in raw_tasks:
                    try:
                        future = pool.submit(_worker_play_game, task)
                    except BaseException as exc:
                        raise ArenaWorkerError(
                            f"Golden Arena worker could not be started for "
                            f"pair={task.pair_id!r} game={task.game_id!r}: "
                            f"{type(exc).__name__}: {exc}",
                            task=task,
                        ) from exc
                    futures[future] = task
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except BaseException as exc:
                        raise ArenaWorkerError(
                            f"Golden Arena worker failed for pair={task.pair_id!r} "
                            f"game={task.game_id!r}: {type(exc).__name__}: {exc}",
                            task=task,
                        ) from exc
                    if record.game_id in by_game_id:
                        raise ArenaWorkerError(
                            f"Worker returned duplicate game_id: {record.game_id!r}",
                            task=task,
                        )
                    by_game_id[record.game_id] = record
        except BaseException:
            self._failed = True
            raise

        try:
            ordered = tuple(by_game_id[task.game_id] for task in raw_tasks)
            for task, record in zip(raw_tasks, ordered):
                _validate_worker_record(record, task)
            self._validate_collected_pairs(raw_tasks, ordered)
            # Recompute only after every requested task has returned and passed
            # validation.  There is no partial summary state.
            if all(sum(task.pair_id == candidate.pair_id for task in raw_tasks) == 2 for candidate in raw_tasks):
                recompute_summary(ordered)
        except BaseException:
            self._failed = True
            raise
        self._records.extend(ordered)
        self._game_ids.update(record.game_id for record in ordered)
        self._pair_ids.update(record.pair_id for record in ordered)
        if all(sum(task.pair_id == candidate.pair_id for task in raw_tasks) == 2 for candidate in raw_tasks):
            self._last_schedule = self._canonical_schedule(self._records)
        else:
            self._last_schedule = ()
        return ordered

    # Short alias for callers that prefer executor terminology.
    run = play_games

    def play_pairs(self, pairs: Iterable[PairTask]) -> tuple[GameRecord, ...]:
        tasks = self.build_tasks(pairs)
        if not tasks:
            return ()
        records = self.play_games(tasks)
        # play_pairs always declares complete pairs; this is an explicit
        # post-collection validation boundary.
        for offset in range(0, len(records), 2):
            validate_pair_records(records[offset : offset + 2])
        recompute_summary(records)
        return records

    def play_pair(
        self,
        *,
        pair_id: str,
        start_state: GoldenState | None = None,
        start_trace: Sequence[int | str] = (),
        game_ids: tuple[str, str] | None = None,
        allow_research_komi: bool = False,
    ) -> tuple[GameRecord, GameRecord]:
        records = self.play_pairs(
            (
                PairTask(
                    pair_id=pair_id,
                    start_state=start_state,
                    start_trace=tuple(start_trace),
                    game_ids=game_ids,
                    allow_research_komi=allow_research_komi,
                ),
            )
        )
        return records[0], records[1]

    def summary(self) -> ArenaSummary:
        if self._failed:
            raise ArenaWorkerError("Process Arena is failed closed; no canonical summary exists")
        if not self._records:
            raise ValueError("Process Arena has no completed pair evidence")
        return recompute_summary(self._records)

    def manifest(self, *, require_canonical: bool = False) -> RunManifest:
        if self._failed:
            raise ArenaWorkerError("Process Arena is failed closed; no canonical manifest exists")
        if not self._records:
            raise ValueError("Cannot build a RunManifest without completed pair evidence")
        summary = recompute_summary(self._records)
        manifest = RunManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            run_id=self.run_id,
            experiment_profile_id=PROFILE_ID,
            experiment_fingerprint=EXPERIMENT_FINGERPRINT,
            git_commit_sha=self.code_identity.git_commit_sha,
            git_tree_sha=self.code_identity.git_tree_sha,
            git_worktree_clean=self.code_identity.working_tree_clean,
            canonical_evidence=self.code_identity.canonical,
            rules_profile_id=RULES_PROFILE_ID,
            rules_fingerprint=self.contract.rules_fingerprint,
            topology_fingerprint=self.contract.topology_fingerprint,
            komi=self.contract.komi,
            arena_contract_id=self.contract.contract_id,
            search_implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_implementation_fingerprint=self._records[0].search_implementation_fingerprint,
            search_contract_fingerprint=self._records[0].search_contract_fingerprint,
            search_settings=self.contract.search.evidence(),
            seed_derivation_id=SEED_DERIVATION_ID,
            master_seed=self.master_seed,
            pair_count=len(summary.paired_results),
            game_count=summary.games,
            pair_schedule=pair_schedule_from_records(self._records),
            player_A=self.player_A_identity,
            player_B=self.player_B_identity,
        )
        from .arena import validate_run_evidence

        validate_run_evidence(manifest, self._records, require_canonical=require_canonical)
        return manifest

    def _validate_tasks(self, tasks: Sequence[GameTask]) -> None:
        ids: set[str] = set(self._game_ids)
        pair_ids: set[str] = set()
        for index, task in enumerate(tasks):
            if task.ordinal != index:
                raise ValueError("GameTask ordinal must equal its parent-declared canonical position")
            if task.game_id in ids:
                raise ValueError(f"Duplicate game_id rejected: {task.game_id}")
            ids.add(task.game_id)
            if not task.game_id or not task.pair_id:
                raise ValueError("GameTask game_id and pair_id must be non-empty")
            if task.pair_id in pair_ids:
                # A pair may occur twice, but not more than twice and not
                # after another pair has started; this keeps the persisted
                # schedule unambiguous.
                prior = [item for item in tasks[:index] if item.pair_id == task.pair_id]
                if len(prior) >= 2:
                    raise ValueError(f"Pair {task.pair_id!r} contains more than two tasks")
            pair_ids.add(task.pair_id)
            if task.black_player not in ("A", "B"):
                raise ValueError("GameTask black_player must be A or B")
            if task.master_seed != self.master_seed or task.run_id != self.run_id:
                raise ValueError("GameTask run identity/master_seed does not match executor")
            if task.contract != self.contract or task.code_identity != self.code_identity:
                raise ValueError("GameTask contract/code identity does not match executor")
            if task.player_A_identity != self.player_A_identity or task.player_B_identity != self.player_B_identity:
                raise ValueError("GameTask player identity does not match executor")
        if len(ids) != len(self._game_ids) + len(tasks):
            raise ValueError("Duplicate game_id rejected")

    @staticmethod
    def _canonical_schedule(records: Sequence[GameRecord]) -> tuple[tuple[str, str, str], ...]:
        grouped: dict[str, dict[str, str]] = {}
        for record in records:
            grouped.setdefault(record.pair_id, {})[record.black_player] = record.game_id
        schedule: list[tuple[str, str, str]] = []
        for pair_id, games in grouped.items():
            if set(games) != {"A", "B"}:
                raise ValueError(f"Pair {pair_id!r} is missing one color-swapped game")
            schedule.append((pair_id, games["A"], games["B"]))
        return tuple(schedule)

    @staticmethod
    def _validate_collected_pairs(tasks: Sequence[GameTask], records: Sequence[GameRecord]) -> None:
        by_pair: dict[str, list[GameRecord]] = {}
        for record in records:
            by_pair.setdefault(record.pair_id, []).append(record)
        task_counts: dict[str, int] = {}
        for task in tasks:
            task_counts[task.pair_id] = task_counts.get(task.pair_id, 0) + 1
        for pair_id, count in task_counts.items():
            if count == 2:
                pair = by_pair[pair_id]
                by_black = {record.black_player: record for record in pair}
                if set(by_black) != {"A", "B"}:
                    raise ValueError(
                        f"Pair {pair_id!r} must contain exactly one A-black and one B-black game"
                    )
                validate_pair_records((by_black["A"], by_black["B"]))
            elif count > 2:
                raise ValueError(f"Pair {pair_id!r} must contain exactly two tasks")


__all__ = [
    "ArenaPairTask",
    "ArenaWorkerError",
    "CheckpointPlayerSpec",
    "GameTask",
    "PairTask",
    "PlayerSpec",
    "ProcessParallelGoldenArena",
]
