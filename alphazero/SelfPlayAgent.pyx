# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools
import time

from alphazero.MCTS import MCTS
from alphazero.envs.gocube.records import reserve_game_id
from alphazero.envs.gocube.selfplay_semantics import (
    CLEANUP_1,
    CLEANUP_2,
    KATAGO_CLEANUP_TRAINING_DEFAULTS,
    apply_pass_would_end_phase_feature,
    rebase_cleanup_training_state,
)
from alphazero.search_contract import KATAGO_PINNED_SEARCH_UTILITY_MODE


def _optional_arg(args, name, default):
    if hasattr(args, 'get'):
        return args.get(name, default)
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


class SelfPlayAgent(mp.Process):
    def __init__(self, id, game_cls, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, pause_event: mp.Event(), args, _is_arena=False, _is_warmup=False,
                 telemetry=None, score_tensor=None, ownership_tensor=None):
        super().__init__()
        self.id = id
        self.game_cls = game_cls
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        if _is_arena:
            self.batch_size = policy_tensor.shape[0]
        else:
            self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.score_tensor = score_tensor
        self.ownership_tensor = ownership_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []
        self.histories = []
        self.temps = []
        self.next_reset = []
        self.mcts = []
        self.games_played = games_played
        self.complete_count = complete_count
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.args = args
        self._is_arena = _is_arena
        self._is_warmup = _is_warmup
        self.telemetry = telemetry
        self.score_aware = (
            _optional_arg(args, 'search_utility_mode', 'legacy') == KATAGO_PINNED_SEARCH_UTILITY_MODE
        )
        if self.score_aware and not _is_warmup and (score_tensor is None or ownership_tensor is None):
            raise ValueError('KataGo-derived SelfPlayAgent requires score_tensor and ownership_tensor')

        cleanup_defaults = KATAGO_CLEANUP_TRAINING_DEFAULTS
        self.cleanup_training_prob = float(_optional_arg(args, 'gocube_cleanup_training_prob', 0.0))
        self.cleanup_training_prelude_area_prop = float(
            _optional_arg(args, 'gocube_cleanup_training_prelude_area_prop', cleanup_defaults['prelude_area_prop'])
        )
        self.cleanup_training_gamma_shape = float(
            _optional_arg(args, 'gocube_cleanup_training_gamma_shape', cleanup_defaults['prelude_gamma_shape'])
        )
        self.cleanup_training_policy_temperature = float(
            _optional_arg(args, 'gocube_cleanup_training_policy_temperature', cleanup_defaults['policy_temperature'])
        )
        if not 0.0 <= self.cleanup_training_prob <= 1.0:
            raise ValueError('gocube_cleanup_training_prob must be within [0,1]')
        if self.cleanup_training_prelude_area_prop < 0.0:
            raise ValueError('gocube_cleanup_training_prelude_area_prop must be non-negative')
        if self.cleanup_training_gamma_shape <= 0.0:
            raise ValueError('gocube_cleanup_training_gamma_shape must be positive')
        if self.cleanup_training_policy_temperature <= 0.0:
            raise ValueError('gocube_cleanup_training_policy_temperature must be positive')

        self.cleanup_training_phase = []
        self.cleanup_training_moves_left = []
        self.cleanup_training_prelude_total = []
        self.cleanup_training_metadata = []
        self.root_policy_cache = []

        self.recording_enabled = bool(
            getattr(args, "gocube_recording_enabled", False) and not _is_arena
        )
        self.record_registry = getattr(args, "gocube_game_id_registry", None)
        self.record_id_prefix = getattr(args, "gocube_game_id_prefix", None)
        self.game_ids = []
        self.move_histories = []
        self.game_start_times = []
        if _is_arena:
            self.player_to_index = list(range(game_cls.num_players()))
            np.random.shuffle(self.player_to_index)
            self.batch_indices = None
        if _is_warmup:
            action_size = game_cls.action_size()
            self._WARMUP_POLICY = torch.full((action_size,), 1 / action_size).to(policy_tensor.device)
            value_size = game_cls.num_players() + 1
            self._WARMUP_VALUE = torch.full((value_size,), 1 / value_size).to(value_tensor.device)
            if score_tensor is not None:
                self._WARMUP_SCORE = torch.zeros((1,)).to(score_tensor.device)
            if ownership_tensor is not None:
                self._WARMUP_OWNERSHIP = torch.full(
                    ownership_tensor.shape[1:], 1.0 / ownership_tensor.shape[-1]
                ).to(ownership_tensor.device)
        self.fast = False
        for _ in range(self.batch_size):
            self.games.append(self.game_cls())
            self.histories.append([])
            self.game_ids.append(None)
            self.move_histories.append([])
            self.game_start_times.append(None)
            self.temps.append(self.args.startTemp)
            self.next_reset.append(0)
            self.mcts.append(self._get_mcts())
            self.cleanup_training_phase.append(None)
            self.cleanup_training_moves_left.append(0)
            self.cleanup_training_prelude_total.append(0)
            self.cleanup_training_metadata.append(None)
            self.root_policy_cache.append(None)

    def _get_mcts(self):
        if self._is_arena:
            return tuple([MCTS(self.args) for _ in range(self.game_cls.num_players())])
        return MCTS(self.args)

    def _mcts(self, index: int) -> MCTS:
        mcts = self.mcts[index]
        if self._is_arena:
            return mcts[self.games[index].player]
        return mcts

    def _check_pause(self):
        while self.pause_event.is_set():
            time.sleep(.1)

    def _telemetry_add(self, key, amount=1):
        if self.telemetry is None:
            return
        counter = self.telemetry.get(key)
        if counter is None:
            return
        lock = counter.get_lock()
        lock.acquire()
        try:
            counter.value += amount
        finally:
            lock.release()

    def _cleanup_training_active(self, index):
        phases = getattr(self, 'cleanup_training_phase', None)
        return phases is not None and index < len(phases) and phases[index] is not None

    def _cleanup_slot_set(self, name, index, value):
        slots = getattr(self, name, None)
        if slots is not None and index < len(slots):
            slots[index] = value

    def _cancel_cleanup_training_plan(self, index):
        self._cleanup_slot_set('cleanup_training_phase', index, None)
        self._cleanup_slot_set('cleanup_training_moves_left', index, 0)
        self._cleanup_slot_set('cleanup_training_prelude_total', index, 0)
        self._cleanup_slot_set('root_policy_cache', index, None)

    def _start_cleanup_training(self, index):
        phases = getattr(self, 'cleanup_training_phase', None)
        if phases is None or index >= len(phases):
            return False
        phase = phases[index]
        if phase not in (CLEANUP_1, CLEANUP_2):
            return False
        game = self.games[index]
        state = getattr(game, 'semantic_state', None)
        if state is None or getattr(state, 'phase', None) != 'main' or getattr(state, 'terminal_kind', None) is not None:
            self._cancel_cleanup_training_plan(index)
            return False

        rebased_state = rebase_cleanup_training_state(state, phase)
        self.games[index] = self.game_cls(rebased_state)
        self.histories[index] = []
        self.temps[index] = self.args.startTemp
        self.mcts[index] = self._get_mcts()
        self.next_reset[index] = 0
        self._cleanup_slot_set('root_policy_cache', index, None)
        prelude_totals = getattr(self, 'cleanup_training_prelude_total', None)
        prelude_total = prelude_totals[index] if prelude_totals is not None and index < len(prelude_totals) else 0
        metadata = {
            'mode': 'cleanup_training',
            'began_in_phase': phase,
            'prelude_moves': int(prelude_total),
            'initial_player': int(rebased_state.current_player),
            'initial_board': [int(v) for v in np.asarray(rebased_state.board).reshape(-1)],
        }
        self._cleanup_slot_set('cleanup_training_metadata', index, metadata)
        self._cleanup_slot_set('cleanup_training_phase', index, None)
        self._cleanup_slot_set('cleanup_training_moves_left', index, 0)
        if getattr(self, 'recording_enabled', False):
            self.game_ids[index] = None
            self.move_histories[index] = []
            self.game_start_times[index] = None
        return True

    def _sample_cleanup_training_plan(self, index):
        phases = getattr(self, 'cleanup_training_phase', None)
        if phases is None or index >= len(phases):
            return
        self._cancel_cleanup_training_plan(index)
        self._cleanup_slot_set('cleanup_training_metadata', index, None)
        if (
            not getattr(self, 'score_aware', False)
            or self._is_arena
            or self._is_warmup
            or self.cleanup_training_prob <= 0.0
            or np.random.random_sample() >= self.cleanup_training_prob
        ):
            return
        state = getattr(self.games[index], 'semantic_state', None)
        if state is None or getattr(state, 'phase', None) != 'main':
            return

        phase = CLEANUP_1 if np.random.random_sample() < 0.5 else CLEANUP_2
        point_count = int(self.game_cls.logical_topology().point_count)
        mean = float(point_count) * self.cleanup_training_prelude_area_prop
        if mean <= 0.0:
            moves = 0
        else:
            moves = int(np.floor(np.random.gamma(
                self.cleanup_training_gamma_shape,
                mean / self.cleanup_training_gamma_shape,
            )))
        self._cleanup_slot_set('cleanup_training_phase', index, phase)
        self._cleanup_slot_set('cleanup_training_moves_left', index, moves)
        self._cleanup_slot_set('cleanup_training_prelude_total', index, moves)
        if moves <= 0:
            self._start_cleanup_training(index)

    def _cleanup_prelude_policy(self, index, fallback_policy):
        caches = getattr(self, 'root_policy_cache', None)
        raw = caches[index] if caches is not None and index < len(caches) else None
        if raw is None:
            raw = fallback_policy
        policy = np.asarray(raw, dtype=np.float64).reshape(-1).copy()
        valid = np.asarray(self.games[index].valid_moves(), dtype=np.uint8).reshape(-1)
        if policy.size != valid.size:
            self._cancel_cleanup_training_plan(index)
            return None
        policy[valid == 0] = 0.0
        policy[policy < 0.0] = 0.0
        if float(policy.sum()) <= 0.0:
            self._cancel_cleanup_training_plan(index)
            return None
        inv_temp = 1.0 / self.cleanup_training_policy_temperature
        policy = np.power(policy, inv_temp)
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0.0:
            self._cancel_cleanup_training_plan(index)
            return None
        return policy / total

    def _select_search_sims(self):
        if self._is_arena:
            self.fast = False
            if hasattr(self.args, 'get'):
                return self.args.get('arenaMCTSSims', self.args.numMCTSSims)
            return getattr(self.args, 'arenaMCTSSims', self.args.numMCTSSims)

        self.fast = np.random.random_sample() < self.args.probFastSim
        return self.args.numFastSims if self.fast else self.args.numMCTSSims \
            if not self._is_warmup else self.args.numWarmupSims

    def run(self):
        try:
            np.random.seed()
            for i in range(self.batch_size):
                self._sample_cleanup_training_plan(i)
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration:
                self._check_pause()
                sims = self._select_search_sims()
                for _ in range(sims):
                    if self.stop_event.is_set(): break
                    self.generateBatch()
                    if self.stop_event.is_set(): break
                    self.processBatch()
                if self.stop_event.is_set(): break
                self.playMoves()
            with self.complete_count.get_lock():
                self.complete_count.value += 1
            if not self._is_arena:
                self.output_queue.close()
                self.output_queue.join_thread()
        except Exception:
            print(traceback.format_exc())

    def generateBatch(self):
        if self._is_arena:
            batch_tensor = [[] for _ in range(self.game_cls.num_players())]
            self.batch_indices = [[] for _ in range(self.game_cls.num_players())]
        for i in range(self.batch_size):
            self._check_pause()
            state = self._mcts(i).find_leaf(self.games[i])
            if self._is_warmup:
                self.policy_tensor[i].copy_(self._WARMUP_POLICY)
                self.value_tensor[i].copy_(self._WARMUP_VALUE)
                if self.score_tensor is not None:
                    self.score_tensor[i].copy_(self._WARMUP_SCORE)
                if self.ownership_tensor is not None:
                    self.ownership_tensor[i].copy_(self._WARMUP_OWNERSHIP)
                continue
            observation = self._mcts(i).search_observation(state) if getattr(self, 'score_aware', False) else state.observation()
            if getattr(self, 'score_aware', False):
                observation = apply_pass_would_end_phase_feature(state, observation)
            data = torch.from_numpy(observation)
            if self._is_arena:
                data = data.view(-1, *state.observation_size())
                player = self.player_to_index[self.games[i].player]
                batch_tensor[player].append(data)
                self.batch_indices[player].append(i)
            else:
                self.batch_tensor[i].copy_(data)
        if self._is_arena:
            for player in range(self.game_cls.num_players()):
                player = self.player_to_index[player]
                data = batch_tensor[player]
                if data:
                    batch_tensor[player] = torch.cat(data)
            self.output_queue.put(batch_tensor)
            self.batch_indices = list(itertools.chain.from_iterable(self.batch_indices))
        if not self._is_warmup:
            self.ready_queue.put(self.id)

    def processBatch(self):
        if not self._is_warmup:
            self.batch_ready.wait()
            self.batch_ready.clear()
        for i in range(self.batch_size):
            self._check_pause()
            index = self.batch_indices[i] if self._is_arena else i
            if getattr(self, 'score_aware', False):
                if self._cleanup_training_active(i) and self._mcts(i).depth == 0:
                    self._cleanup_slot_set(
                        'root_policy_cache',
                        i,
                        np.array(self.policy_tensor[index].data.numpy(), dtype=np.float64, copy=True),
                    )
                self._mcts(i).process_search_results(
                    self.games[i],
                    self.value_tensor[index].data.numpy(),
                    self.policy_tensor[index].data.numpy(),
                    self.score_tensor[index].data.numpy(),
                    self.ownership_tensor[index].data.numpy(),
                    False if self._is_arena else self.args.add_root_noise,
                    False if self._is_arena else self.args.add_root_temp,
                )
            else:
                self._mcts(i).process_results(
                    self.games[i],
                    self.value_tensor[index].data.numpy(),
                    self.policy_tensor[index].data.numpy(),
                    False if self._is_arena else self.args.add_root_noise,
                    False if self._is_arena else self.args.add_root_temp
                )

    def playMoves(self):
        recording_enabled = getattr(self, "recording_enabled", False)
        for i in range(self.batch_size):
            self._check_pause()
            self.temps[i] = self.args.temp_scaling_fn(
                self.temps[i], self.games[i].turns, self.game_cls.max_turns()
            ) if not self._is_arena else self.args.arenaTemp
            policy = self._mcts(i).probs(self.games[i], self.temps[i])
            in_cleanup_prelude = self._cleanup_training_active(i)
            if in_cleanup_prelude:
                prelude_policy = self._cleanup_prelude_policy(i, policy)
                if prelude_policy is None:
                    in_cleanup_prelude = False
                else:
                    policy = prelude_policy
            action = np.random.choice(self.games[i].action_size(), p=policy)
            if not self._is_arena and not in_cleanup_prelude:
                self._telemetry_add('fast_decisions' if self.fast else 'regular_decisions')
            if not self.fast and not self._is_arena and not in_cleanup_prelude:
                self.histories[i].append((self.games[i].clone(), self._mcts(i).probs(self.games[i])))
            if recording_enabled and not in_cleanup_prelude:
                state = getattr(self.games[i], "semantic_state", None)
                self.game_start_times[i] = self.game_start_times[i] or time.time()
                move_record = {
                    "move_number": len(self.move_histories[i]) + 1,
                    "player": "black" if self.games[i].player == 0 else "white",
                    "phase": getattr(state, "phase", None),
                    "action": int(action),
                    "move": "PASS" if action == self.game_cls.pass_action() else self.game_cls.point_id_for_action(int(action)),
                }
                metadata_slots = getattr(self, 'cleanup_training_metadata', None)
                metadata = metadata_slots[i] if metadata_slots is not None and i < len(metadata_slots) else None
                if not self.move_histories[i] and metadata is not None:
                    move_record["training_start"] = metadata
                self.move_histories[i].append(move_record)
            if self._is_arena:
                [mcts.update_root(self.games[i], action) for mcts in self.mcts[i]]
            else:
                self._mcts(i).update_root(self.games[i], action)
            self.games[i].play_action(action)
            self._cleanup_slot_set('root_policy_cache', i, None)

            if in_cleanup_prelude:
                moves_left_slots = getattr(self, 'cleanup_training_moves_left', None)
                moves_left = 0
                if moves_left_slots is not None and i < len(moves_left_slots):
                    moves_left = max(0, moves_left_slots[i] - 1)
                    moves_left_slots[i] = moves_left
                state = getattr(self.games[i], 'semantic_state', None)
                if (
                    state is not None
                    and getattr(state, 'terminal_kind', None) is None
                    and getattr(state, 'phase', None) == 'main'
                    and moves_left <= 0
                ):
                    if self._start_cleanup_training(i):
                        continue
                elif state is None or getattr(state, 'terminal_kind', None) is not None or getattr(state, 'phase', None) != 'main':
                    self._cancel_cleanup_training_plan(i)

            if self.args.mctsResetThreshold and self.games[i].turns >= self.next_reset[i]:
                self.mcts[i] = self._get_mcts()
                self.next_reset[i] = self.games[i].turns + self.args.mctsResetThreshold
            winstate = self.games[i].win_state()
            if not winstate.any():
                continue

            final_game = self.games[i].clone()
            if recording_enabled:
                lock = self.games_played.get_lock()
                lock.acquire()
                accepted = self.games_played.value < self.args.gamesPerIteration
                if accepted:
                    self.games_played.value += 1
                    game_number = self.games_played.value
                lock.release()
                if not accepted:
                    continue
                self.game_ids[i] = reserve_game_id(self.record_registry, self.record_id_prefix)
                self.result_queue.put((
                    final_game, winstate, self.id, {
                        "game_id": self.game_ids[i],
                        "moves": self.move_histories[i],
                        "start_time": self.game_start_times[i] or time.time(),
                        "end_time": time.time(),
                        "game_number_inside_iteration": game_number,
                    }
                ))
            else:
                self.result_queue.put((final_game, winstate, self.id))
                lock = self.games_played.get_lock()
                lock.acquire()
                accepted = self.games_played.value < self.args.gamesPerIteration
                if accepted:
                    self.games_played.value += 1
                lock.release()
                if not accepted:
                    continue

            if not self._is_arena:
                training_valid = True
                if hasattr(final_game, "has_training_result"):
                    training_valid = final_game.has_training_result()
                if training_valid:
                    auxiliary = bool(getattr(self.args, "gocube_auxiliary_targets", False))
                    ownership_mask = None
                    if auxiliary:
                        targets = final_game.training_targets()
                        if len(targets) == 3:
                            score_target, ownership_target, ownership_mask = targets
                        else:
                            score_target, ownership_target = targets
                    for hist in self.histories[i]:
                        self._check_pause()
                        self._telemetry_add('base_positions')
                        is_endgame = bool(
                            hasattr(hist[0], "is_endgame_training_state")
                            and hist[0].is_endgame_training_state()
                        )
                        if is_endgame:
                            self._telemetry_add('base_endgame_positions')
                        data = hist[0].symmetries(hist[1]) if self.args.symmetricSamples else ((hist[0], hist[1]),)
                        repeat = 1
                        endgame_weight = int(getattr(self.args, "gocube_endgame_sample_weight", 1))
                        if endgame_weight > 1 and is_endgame:
                            repeat = endgame_weight
                        for state, pi in data:
                            self._check_pause()
                            observation = state.observation()
                            if getattr(self, 'score_aware', False):
                                observation = apply_pass_would_end_phase_feature(state, observation)
                            sample = (observation, pi, np.array(winstate, dtype=np.float32))
                            if auxiliary:
                                sample = sample + (score_target, ownership_target)
                                if ownership_mask is not None:
                                    sample = sample + (ownership_mask,)
                            for _ in range(repeat):
                                self.output_queue.put(sample)
                            if repeat > 1:
                                self._telemetry_add('endgame_extra_samples', repeat - 1)

            self.games[i] = self.game_cls()
            self.histories[i] = []
            if recording_enabled:
                self.game_ids[i] = None
                self.move_histories[i] = []
                self.game_start_times[i] = None
            self.temps[i] = self.args.startTemp
            self.mcts[i] = self._get_mcts()
            self._cleanup_slot_set('cleanup_training_metadata', i, None)
            self._cleanup_slot_set('root_policy_cache', i, None)
            if getattr(self, 'cleanup_training_phase', None) is not None:
                self._sample_cleanup_training_plan(i)
