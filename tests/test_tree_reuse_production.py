"""Production wiring checks use mocked parents/transports and temporary data."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gocube_golden import torus9_monolith as core, torus9_selfplay as sp
from gocube_golden.artifact_graph import EffectiveConfig
from gocube_golden.orchestrator_v2 import operator_job as job, production_entrypoint as entry
from gocube_golden.orchestrator_v2.run_spec import _validate_arena_config_payload
from gocube_golden.orchestrator_v2.topology_binding import get_topology_binding
from gocube_golden.provenance import CodeIdentity
from gocube_golden.search import Evaluation, SearchEvaluationRequest, SequentialPUCTSession
from gocube_golden.selfplay_engine import InferenceNeed
from gocube_golden.torus9_adaptation import AdaptationSearchContract, PCRSearchContract
from gocube_golden.torus9_pcr import PlayoutCapRandomization
from tools.arena_profiles import get_profile
from test_orchestrator_operator_job import parent, parameters


@pytest.mark.parametrize('reuse', [False, True])
def test_operator_compilation_wires_selfplay_and_arena_to_every_training_arm(parent, reuse):
    raw = parameters()
    raw['self_play'] = {'tree_reuse': reuse}
    raw['arena']['tree_reuse'] = reuse
    normalized = job.parse_job(raw)
    assert job.parse_job(normalized) == normalized
    steps = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps']
    configs = [steps[0]['config']['effective_config']]
    configs.extend(arm['config'] for arm in steps[1]['config']['arms'])
    for config in configs:
        assert config['self_play']['tree_reuse'] is reuse
        assert config['arena']['tree_reuse'] is reuse
        profile_id = get_topology_binding('torus9').default_arena_profile(EffectiveConfig.from_dict(config))
        assert get_profile(profile_id).tree_reuse is reuse
    assert get_profile(steps[0]['config']['arena_profile']).tree_reuse is reuse
    assert get_profile(steps[1]['config']['arena']['profile']).tree_reuse is reuse


def test_new_job_requires_explicit_opt_in_even_if_parent_enabled_reuse(parent):
    inherited = parent.effective_config.config.to_dict()
    inherited['self_play']['tree_reuse'] = True
    parent.effective_config.config = EffectiveConfig.from_dict(inherited)
    compiled = job.compile_job(parameters(), resolver=SimpleNamespace())
    cfg = compiled['workflow']['steps'][0]['config']['effective_config']
    assert cfg['self_play'].get('tree_reuse', False) is False
    assert cfg['arena'].get('tree_reuse', False) is False
    assert parent.effective_config.config.self_play['tree_reuse'] is True


def test_standalone_arena_and_winner_selection_inherit_or_override_reuse():
    raw = {'schema': job.SCHEMA, 'run_id': 'synthetic-arena', 'arena': {'tree_reuse': True},
           'arenas': [{'id': 'on', 'candidate': 'source/M1', 'reference': 'source/M0'},
                      {'id': 'off', 'candidate': 'source/M1', 'reference': 'source/M0', 'tree_reuse': False}]}
    normalized = job.parse_job(raw)
    assert [a['tree_reuse'] for a in normalized['arenas']] == [True, False]
    for item in normalized['arenas']:
        assert get_profile(job._arena_profile(item)).tree_reuse is item['tree_reuse']
    raw = parameters()
    raw.pop('parent')
    raw.pop('ab_tests')
    raw['arena']['tree_reuse'] = True
    raw['winner_selection'] = {'candidate': 'source/M1', 'reference': 'source/M0'}
    assert job.parse_job(raw)['winner_selection']['tree_reuse'] is True


@pytest.mark.parametrize('field', ['self_play', 'arena'])
@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_operator_rejects_non_boolean_switch_before_resolving_any_artifact(monkeypatch, field, value):
    monkeypatch.setattr(job, 'resolve_parent', lambda *a, **kw: pytest.fail('resolved invalid config'))
    raw = parameters()
    raw[field] = {**raw.get(field, {}), 'tree_reuse': value}
    with pytest.raises(ValueError, match='boolean'):
        job.compile_job(raw, resolver=SimpleNamespace())


@pytest.mark.parametrize('reuse', [False, True])
def test_v2_arena_search_config_reaches_torus_profile_and_callbacks(monkeypatch, reuse):
    search = {'tree_reuse': reuse, 'simulations': 8}
    _validate_arena_config_payload({'search': search})
    profile_id = entry._torus_profile_with_search('torus9|komi=1.5|5ch', search)
    profile = get_profile(profile_id)
    seen = []
    import tools.arena_profiles.torus9 as torus
    monkeypatch.setattr(torus, 'run_cooperative_arena_worker', lambda **kwargs: seen.append(kwargs['callbacks']))
    profile.worker_main(0, None, 1, 0., 'candidate', 'reference', None, None, None, None, None, None)
    settings = seen[0].search_settings
    assert settings.tree_reuse is reuse and settings.simulations == 8
    assert settings.cpuct == 1.25 and settings.fpu == 0.
    assert get_profile(entry._torus_profile_with_search(profile_id, {})).tree_reuse is reuse
    assert get_profile(entry._torus_profile_with_search(profile_id, {'tree_reuse': False})).tree_reuse is False


@pytest.mark.parametrize('mode', ['fixed', 'pcr'])
@pytest.mark.parametrize('reuse', [False, True])
def test_selfplay_uses_sampled_action_and_retains_per_game_tree(monkeypatch, mode, reuse):
    caps = PlayoutCapRandomization(4, 8, .5)
    contract = (AdaptationSearchContract(simulations=8, komi=1.5, tree_reuse=reuse) if mode == 'fixed'
                else PCRSearchContract(simulations=8, komi=1.5, tree_reuse=reuse, pcr=caps))
    context = sp.Torus9SelfPlayWorkerContext(run_id='synthetic-reuse', model_checkpoint_label='M0',
        checkpoint_artifact_hash='sha256:synthetic', master_seed=1, profile_fingerprint='synthetic',
        code_identity=CodeIdentity('a'*40, 'b'*40, True), contract=contract,
        profile_id=core.TORUS9_CURRENT_PROFILE_ID, expected_model_hash='synthetic')
    game, other = sp._Torus9CooperativeGame(context, 'g1', None), sp._Torus9CooperativeGame(context, 'g2', None)
    assert game._tree is not other._tree
    game._start_search()
    session = game._session
    assert session.settings.tree_reuse is reuse
    evaluator_policy = (1.,) + (0.,)*81
    while isinstance(session.advance(), SearchEvaluationRequest):
        session.resume(Evaluation(evaluator_policy, (0., 1., 0.)))
    result = session.result
    # Choose a different legal action than argmax: reroot must follow sampling.
    sampled = next(a for a in result.legal_actions if a != result.action)
    child = game._tree.root.edges[sampled].child if reuse else None
    monkeypatch.setattr(sp, 'sample_action_from_search_result', lambda *a, **kw: sampled)
    need = game.advance()
    assert isinstance(need, InferenceNeed)
    assert game.trace == [sampled] and game._session.tree is game._tree
    if reuse and child is not None:
        assert game._tree.root is child
    if reuse:
        assert game._tree.root.state.state_key == game.state.state_key
    else:
        assert game._tree.root is None
    assert other._tree.root is None
    off = replace(contract, tree_reuse=False)
    assert (off.fingerprint != contract.fingerprint) is reuse


def test_cube_reuse_wiring_includes_complete_observation_history():
    from gocube_golden.cube_arena_contract_v2 import CubeArenaSearchConfig
    from gocube_golden.cube_family import initial_cube_state
    from gocube_golden.cube_observation_v2 import initial_cube_observation_context
    from gocube_golden.cube_search import CubeSearchAdapter, CubeSearchPosition
    from gocube_golden.cube_selfplay_contract import CubeSelfPlaySearchContract
    from gocube_golden.search import SearchTree
    from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile
    from gocube_golden.orchestrator_v2.cube_production import _selfplay_plan
    from gocube_golden.orchestrator_v2.topology_binding import _cube_arena_search
    search_config = CubeSelfPlaySearchContract(tree_reuse=True)
    config = EffectiveConfig(topology='cube2', compatibility={'size': 2},
        self_play={**search_config.concrete_search_config_identity(), 'games_per_iteration': 2},
        arena={'simulations': 4, 'cpuct': 1.25, 'fpu': 0., 'watchdog': 100, 'tree_reuse': True})
    resolved = SimpleNamespace(effective_config=SimpleNamespace(config=config))
    assert _selfplay_plan(resolved).search.tree_reuse is True
    assert _cube_arena_search(config).tree_reuse is True
    arena = CubeArenaSearchConfig(simulations=4, cpuct=1.25, fpu=0., watchdog=100, tree_reuse=True)
    profile = CubeV2ArenaProfile(size=2, search_config=arena)
    assert get_profile(profile.profile_id).search_config.tree_reuse is True
    assert arena.search_settings.tree_reuse is True
    assert CubeSelfPlaySearchContract(tree_reuse=True).puct_settings.tree_reuse is True
    assert arena.fingerprint != replace(arena, tree_reuse=False).fingerprint
    state = initial_cube_state(size=2)
    position = CubeSearchPosition(state, initial_cube_observation_context(state))
    tree = SearchTree(True)
    adapter = CubeSearchAdapter()
    session = SequentialPUCTSession(position, arena.search_settings, tree=tree, adapter=adapter)
    while isinstance(session.advance(), SearchEvaluationRequest):
        session.resume(Evaluation((0.,)*24 + (1.,), (0., 1., 0.)))
    after = adapter.apply_action(position, core.PASS)
    child = tree.root.edges[core.PASS].child
    assert tree.advance(core.PASS, after) and tree.root is child
    wrong_context = replace(after.observation_context, previous_action=None)
    wrong = CubeSearchPosition(after.game_state, wrong_context)
    assert wrong.game_state.state_key == after.game_state.state_key
    assert wrong.state_key != after.state_key
    assert tree.prepare_root(wrong) is not child
