"""No production computation may enter through an unowned Python call."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from gocube_golden.orchestrator_v2.execution_permit import (
    PERMIT_ENV, PERMIT_KEY_ENV, _production_authority, _child_execution_permit,
    require_engine_execution,
)


def entrypoints():
    from selfplay_engine import SelfPlayEngine
    from training_engine import TrainingEngine
    from gocube_golden.selfplay_engine import run_cooperative_selfplay
    from gocube_golden.torus9_selfplay import run_torus9_selfplay_games
    from gocube_golden.cube_selfplay_v2 import run_cube_selfplay_games
    from gocube_golden.cube_training_v2 import CubeTrainingAdapter
    from gocube_golden.torus9_training import Torus9TrainingAdapter
    from gocube_golden.torus9_adaptation import AdaptationTrainer, selfplay
    from gocube_golden.torus9_monolith import _Torus9TrainingCore, Torus9OwnershipTrainer, Torus9OwnershipScoreTrainer
    return [
        lambda: SelfPlayEngine.run(None, [], worker_play=None, worker_context=None, infer_batch=None),
        lambda: TrainingEngine.run_iteration(None, state=None, generation=1, output_dir=Path('MUST_NOT_EXIST'), run_id='x', training_seed=0),
        lambda: run_cooperative_selfplay([], adapter=None, engine_config=None),
        lambda: run_torus9_selfplay_games(None, run_id='x', label='x', artifact='x', master_seed=0, profile_fp='x', game_ids=[]),
        lambda: run_cube_selfplay_games(None, []),
        lambda: CubeTrainingAdapter.train(None, None, [], 0),
        lambda: Torus9TrainingAdapter.train(None, None, [], 0),
        lambda: AdaptationTrainer.step(None, []),
        lambda: selfplay(None, checkpoint=None, run_id='x', ids=[], seed=0),
        lambda: _Torus9TrainingCore.train_fixed_budget(None, [], seed=0),
        lambda: Torus9OwnershipTrainer.train_fixed_budget(None, [], seed=0),
        lambda: Torus9OwnershipScoreTrainer.train_fixed_budget(None, [], seed=0),
    ]


@pytest.mark.parametrize('index', range(12))
def test_direct_engines_fail_before_touching_inputs(monkeypatch, tmp_path, index):
    monkeypatch.delenv(PERMIT_ENV, raising=False)
    monkeypatch.delenv(PERMIT_KEY_ENV, raising=False)
    monkeypatch.setenv('AZ_ORCHESTRATOR_VERSION', 'V2')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match='execution permit'):
        entrypoints()[index]()
    assert list(tmp_path.iterdir()) == []


def test_arena_authority_cannot_train_or_selfplay():
    with _production_authority(mode='arena', topology='torus9', run_id='x', code_identity='x'):
        for action in ('training', 'selfplay'):
            with pytest.raises(RuntimeError, match='does not authorize'):
                require_engine_execution('test', action=action)


@pytest.mark.parametrize('topology', ['torus9', 'cube4', 'cube5'])
@pytest.mark.parametrize('permit_action', ['generation', 'arena'])
def test_supervised_generation_permits_both_engines(tmp_path, topology, permit_action):
    script = tmp_path / 'child.py'
    script.write_text('''from gocube_golden.orchestrator_v2.execution_permit import require_engine_execution
import sys
for action in ('training', 'selfplay'):
    require_engine_execution('test-child', action=action, topology=sys.argv[1])
print('AUTHORIZED')
''')
    with _production_authority(mode='continuous', topology=topology, run_id='x', code_identity='x'):
        with _child_execution_permit(action_type=permit_action, topology=topology, run_id='x', code_identity='x'):
            result = subprocess.run([sys.executable, str(script), topology], env=dict(os.environ, PYTHONPATH=str(Path.cwd())), capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == (permit_action == 'generation'), result.stderr
    if permit_action == 'arena':
        assert 'does not authorize' in result.stderr


def test_live_authority_is_topology_scoped():
    with _production_authority(mode='continuous', topology='cube4', run_id='x', code_identity='x'):
        require_engine_execution('test', action='training', topology='cube4')
        with pytest.raises(RuntimeError, match='topology mismatch'):
            require_engine_execution('test', action='selfplay', topology='torus9')


def test_only_existing_orchestrator_modules_mint_execution_authority():
    import ast
    root = Path(__file__).resolve().parents[1]
    allowed = {
        'gocube_golden/orchestrator_v2/execution_permit.py',
        'gocube_golden/orchestrator_v2/production_entrypoint.py',
        'gocube_golden/orchestrator_v2/production_generation.py',
        'gocube_golden/orchestrator_v2/arena_runner.py',
        'gocube_golden/orchestrator_v2/_arena_runner_core.py',
    }
    private = {'_production_authority', '_child_execution_permit', '_test_authority'}
    violations = []
    files = [*root.glob('*.py'), *(root / 'gocube_golden').rglob('*.py'), *(root / 'tools').rglob('*.py')]
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and any(alias.name in private for alias in node.names):
                violations.append(relative)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in private:
                violations.append(relative)
    assert not violations, f'Unauthorized permit issuer: {violations}'


def test_standalone_decision_cannot_wake_legacy_training(monkeypatch, tmp_path):
    from gocube_golden.orchestrator_v2.adaptation import write_decision
    monkeypatch.delenv(PERMIT_ENV, raising=False)
    monkeypatch.delenv(PERMIT_KEY_ENV, raising=False)
    for action in ('continue', 'expand'):
        with pytest.raises(RuntimeError, match='execution permit'):
            write_decision(tmp_path, action, 'unused')
    assert not (tmp_path / 'decision.json').exists()
