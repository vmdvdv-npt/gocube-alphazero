# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools
import time

from alphazero.MCTS import MCTS
from alphazero.envs.gocube.records import reserve_game_id
from alphazero.search_contract import GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE


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
            getattr(args, 'search_utility_mode', 'legacy') == GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE
        )
        if self.score_aware and not _is_warmup and (score_tensor is None or ownership_tensor is None):
            raise ValueError('score-aware SelfPlayAgent requires score_tensor and ownership_tensor')
        self.validation_only = bool(getattr(args, "gocube_validation_only", False) and not _is_arena)
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
            self._WARMUP_VALUE = torch.full((value_size,), 1 / value_size).to(policy_tensor.device)
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
            data = torch.from_numpy(state.observation())
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
            if self.score_aware:
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
                    False if self._is_arena else self.args.add_root_temp,
                )

    def _phase_bucket(self, game):
        state = getattr(game, 'semantic_state', None)
        if state is None:
            return 'main'
        phase = getattr(state, 'phase', 'main')
        if phase == 'main' and int(getattr(state, 'consecutive_passes', 0)) == 1:
            return 'main_after_one_pass'
        if phase == 'cleanup1':
            return 'cleanup1'
        if phase == 'cleanup2':
            return 'cleanup2'
        return 'main'

    def _phase_weight(self, bucket):
        if bucket == 'main_after_one_pass':
            return int(getattr(self.args, 'gocube_main_after_pass_weight', 1))
        if bucket == 'cleanup1':
            return int(getattr(self.args, 'gocube_cleanup1_weight', 1))
        if bucket == 'cleanup2':
            return int(getattr(self.args, 'gocube_cleanup2_weight', 1))
        return 1

    def _record_decision_telemetry(self, game, action):
        state = getattr(game, 'semantic_state', None)
        if state is None:
            return
        phase = getattr(state, 'phase', None)
        player = int(game.player)
        is_pass = int(action) == int(self.game_cls.pass_action())
        if phase == 'main':
            phase_key = 'main'
        elif phase == 'cleanup1':
            phase_key = 'cleanup1'
        elif phase == 'cleanup2':
            phase_key = 'cleanup2'
        else:
            return
        self._telemetry_add(f'phase_{phase_key}_decisions')
        self._telemetry_add(f'phase_{phase_key}_passes' if is_pass else f'phase_{phase_key}_placements')
        if is_pass:
            color = 'black' if player == 0 else 'white'
            self._telemetry_add(f'pass_{phase_key}_{color}')
        if phase == 'main' and is_pass and int(getattr(state, 'consecutive_passes', 0)) == 1:
            turn = int(game.turns) + 1
            self._telemetry_add('main_second_pass_count')
            self._telemetry_add('main_second_pass_turn_sum', turn)
            if turn <= 2:
                self._telemetry_add('main_double_pass_within_2')
            if turn <= 4:
                self._telemetry_add('main_double_pass_within_4')
            if turn <= 8:
                self._telemetry_add('main_double_pass_within_8')

    def _record_search_audit(self, game, action):
        if not self.score_aware:
            return
        state = getattr(game, 'semantic_state', None)
        if state is None or getattr(state, 'phase', None) != 'main' or int(getattr(state, 'consecutive_passes', 0)) != 1:
            return
        sample_probability = float(getattr(self.args, 'gocube_search_audit_probability', 0.25))
        if sample_probability < 1.0 and np.random.random_sample() >= sample_probability:
            return
        diagnostic = self._mcts_for_diagnostic(game)
        if not diagnostic:
            return
        self._telemetry_add('search_audited_positions')
        self._telemetry_add('search_pass_root_prior_sum', diagnostic['pass_root_prior'])
        self._telemetry_add('search_pass_visit_fraction_sum', diagnostic['pass_visit_fraction'])
        self._telemetry_add('search_pass_win_utility_sum', diagnostic['pass_win_utility'])
        self._telemetry_add('search_pass_score_utility_sum', diagnostic['pass_score_utility'])
        self._telemetry_add('search_pass_combined_utility_sum', diagnostic['pass_combined_utility'])
        self._telemetry_add('search_best_nonpass_score_gain_sum', diagnostic['best_nonpass_score_gain'])
        self._telemetry_add('search_best_nonpass_win_delta_sum', diagnostic['best_nonpass_win_delta'])
        # best_action() already applies conditional PASS suppression. The audit
        # must count the pre-suppression score-dominance signal itself, or the
        # guard would systematically miss exactly the cases that were corrected.
        if diagnostic['score_dominated_pass']:
            self._telemetry_add('search_score_dominated_pass')

    @property
    def _mcts_current(self):
        # Set transiently by playMoves before calling _record_search_audit.
        return self.__mcts_current

    def _mcts_for_diagnostic(self, game):
        try:
            return self._mcts_current.pass_diagnostic(game)
        except Exception:
            return {}

    def playMoves(self):
        recording_enabled = getattr(self, "recording_enabled", False)
        for i in range(self.batch_size):
            self._check_pause()
            self.temps[i] = self.args.temp_scaling_fn(
                self.temps[i], self.games[i].turns, self.game_cls.max_turns()
            ) if not self._is_arena else self.args.arenaTemp
            current_mcts = self._mcts(i)
            self.__mcts_current = current_mcts
            policy = current_mcts.probs(self.games[i], self.temps[i])
            action = np.random.choice(self.games[i].action_size(), p=policy)
            if not self._is_arena:
                self._record_decision_telemetry(self.games[i], action)
                self._record_search_audit(self.games[i], action)
                self._telemetry_add('fast_decisions' if self.fast else 'regular_decisions')
            if not self.fast and not self._is_arena and not self.validation_only:
                self.histories[i].append((self.games[i].clone(), current_mcts.probs(self.games[i])))
            if recording_enabled:
                state = getattr(self.games[i], "semantic_state", None)
                self.game_start_times[i] = self.game_start_times[i] or time.time()
                self.move_histories[i].append({
                    "move_number": len(self.move_histories[i]) + 1,
                    "player": "black" if self.games[i].player == 0 else "white",
                    "phase": getattr(state, "phase", None),
                    "action": int(action),
                    "move": "PASS" if action == self.game_cls.pass_action() else self.game_cls.point_id_for_action(int(action)),
                })
            if self._is_arena:
                [mcts.update_root(self.games[i], action) for mcts in self.mcts[i]]
            else:
                current_mcts.update_root(self.games[i], action)
            self.games[i].play_action(action)
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

            if not self._is_arena and not self.validation_only:
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
                        bucket = self._phase_bucket(hist[0])
                        is_endgame = bucket != 'main'
                        if is_endgame:
                            self._telemetry_add('base_endgame_positions')
                        data = hist[0].symmetries(hist[1]) if self.args.symmetricSamples else ((hist[0], hist[1]),)
                        repeat = self._phase_weight(bucket)
                        if repeat < 1:
                            raise ValueError(f'phase sample weight must be at least 1, got {repeat}')
                        for state, pi in data:
                            self._check_pause()
                            self._telemetry_add(f'samples_{bucket}')
                            self._telemetry_add(f'weighted_samples_{bucket}', repeat)
                            sample = (state.observation(), pi, np.array(winstate, dtype=np.float32))
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
