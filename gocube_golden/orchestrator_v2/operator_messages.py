"""Central operator-facing multiline messages for Orchestrator V2."""
from __future__ import annotations


def _pick(mapping: object, *names: str) -> object | None:
    data = dict(mapping or {}) if hasattr(mapping, "items") else {}
    for name in names:
        if data.get(name) is not None:
            return data[name]
    return None


def _line(lines: list[str], label: str, value: object | None, suffix: str = "") -> None:
    if value is not None and value != "":
        lines.append(f"{label}: {value}{suffix}")


def format_training_started(*, topology: str, lineage_id: str, parent_label: str, network: object | None, effective_config: object, arena_cadence: int, arena_config: object) -> str:
    self_play = getattr(effective_config, "self_play", {})
    training = getattr(effective_config, "training", {})
    replay = getattr(effective_config, "replay", {})
    execution = getattr(effective_config, "execution", {})
    arena = getattr(effective_config, "arena", {})
    compatibility = getattr(effective_config, "compatibility", {})
    network = network or _pick(compatibility, "network", "architecture", "architecture_id")
    games = _pick(self_play, "games_per_iteration", "games")
    selfplay_sims = _pick(self_play, "mcts_simulations", "simulations")
    contexts = _pick(execution, "total_active_contexts", "active_contexts", "contexts")
    lr = _pick(training, "learning_rate", "lr")
    steps = _pick(training, "optimizer_steps", "steps")
    batch = _pick(training, "batch_size", "batch")
    replay_generations = _pick(replay, "generations", "window")
    replay_cap = _pick(replay, "cap", "positions_cap", "max_positions")
    arena_sims = _pick(arena, "simulations", "mcts_simulations")
    lines = ["TRAINING STARTED — GoCube AlphaZero", ""]
    _line(lines, "Topology", topology)
    _line(lines, "Lineage", lineage_id)
    _line(lines, "Parent", parent_label)
    _line(lines, "Network", network)
    _line(lines, "Komi", _pick(self_play, "komi"))
    lines.extend(["", "Self-play:"])
    if games is not None:
        lines.append(f"games/generation={games}")
    if selfplay_sims is not None:
        lines.append(f"self-play MCTS={selfplay_sims} sims")
    _line(lines, "Contexts", contexts)
    lines.extend(["", "Training:"])
    if lr is not None:
        lines.append(f"LR={lr}")
    _line(lines, "Steps", steps)
    _line(lines, "Batch", batch)
    lines.extend(["", "Replay:"])
    if replay_generations is not None or replay_cap is not None:
        lines.append(f"replay={replay_generations} generations / {replay_cap} positions")
    lines.extend(["", "Arena:"])
    lines.append(f"Arena cadence=every {arena_cadence} generations")
    _line(lines, "Games", getattr(arena_config, "games", None))
    _line(lines, "MCTS", arena_sims, " sims")
    return "\n".join(lines).rstrip()


def format_arena_started(*, topology: str, evaluation_id: str, candidate: str, reference: str, profile: str, komi: object | None, games: int, simulations: object | None, seed: int, workers: int, contexts: int, batch_cap: int, wait_ms: float) -> str:
    lines = ["ARENA STARTED — GoCube AlphaZero", ""]
    for label, value in (("Topology", topology), ("Evaluation", evaluation_id), ("Candidate", candidate), ("Reference", reference), ("Profile", profile), ("Komi", komi)):
        _line(lines, label, value)
    lines.append("")
    _line(lines, "Games", games)
    _line(lines, "Pairs", games // 2)
    _line(lines, "MCTS", simulations, " sims")
    _line(lines, "Seed", seed)
    lines.append("")
    _line(lines, "Workers", workers)
    _line(lines, "Contexts", contexts)
    _line(lines, "Batch cap", batch_cap)
    _line(lines, "Wait", f"{wait_ms:g}", " ms")
    return "\n".join(lines).rstrip()


def format_action_started(title: str, **fields: object) -> str:
    lines = [f"{title.upper()} — GoCube AlphaZero"]
    for label, value in fields.items():
        _line(lines, label.replace("_", " ").title(), value)
    return "\n".join(lines)


def format_arena_completed(*, topology: str, evaluation_id: str, candidate: str,
                           reference: str, validity: str, wld: tuple[int, int, int],
                           summary: object, execution_code_commit: str | None) -> str:
    data = dict(summary or {})
    telemetry = data.get("telemetry", {})
    lines = ["ARENA COMPLETED — GoCube AlphaZero", ""]
    for label, value in (("Topology", topology), ("Evaluation", evaluation_id),
                         ("Candidate", candidate), ("Reference", reference),
                         ("Validity", validity), ("W/L/D", "/".join(map(str, wld)))):
        _line(lines, label, value)
    _line(lines, "Valid games", _pick(data, "valid_games", "games_valid"))
    _line(lines, "Technical games", _pick(telemetry, "technical_games"))
    _line(lines, "Performance", _pick(telemetry, "performance_status"))
    _line(lines, "Execution commit", execution_code_commit)
    return "\n".join(lines)


__all__ = ["format_action_started", "format_arena_started", "format_arena_completed", "format_training_started"]
