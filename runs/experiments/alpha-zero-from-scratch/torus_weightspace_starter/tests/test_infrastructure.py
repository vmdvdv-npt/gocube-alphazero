import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from model import TorusNet, torus_neighbors
from weights import (PARAMETERS, validate, load_arrays, store_arrays, save_checkpoint,
                     restore_checkpoint, state_digest, sha256)
from experiment_cli import zero_pass, init_session, diagnostic, run_writer
from diagnostics import positions, paired_games, replay
from gocube_golden.rules import (apply_action, legal_actions, group_from_board,
                                 liberties_from_board, IllegalMoveError)
from gocube_golden.state import PASS, BLACK, WHITE, research_state_from_stones, initial_state
from gocube_golden.topology import TORUS_9X9
from gocube_golden.scoring import score_terminal
from gocube_golden.search import wdl_to_side_to_move_utility
from lock_reference import freeze

# CPU-only, synthetic fixtures. No production games or checkpoints in tests.
torch.set_num_threads(1)


class WeightTests(unittest.TestCase):
    def test_zero_constructor_and_schema(self):
        m = TorusNet()
        self.assertEqual(validate(m)['parameter_count'], PARAMETERS)
        self.assertTrue(all(p.dtype == torch.float32 and not p.count_nonzero() for p in m.parameters()))

    def test_numpy_writes_and_rejection(self):
        m = TorusNet()
        arrays = load_arrays(m)
        arrays['input_projection.weight'][:] = np.arange(600,dtype=np.float32).reshape(120,5)/600
        store_arrays(m, arrays)
        self.assertEqual(float(m.input_projection.weight[0,1]), float(arrays['input_projection.weight'][0,1]))
        arrays['pass_policy.bias'][0] = np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            store_arrays(m, arrays)
        arrays.pop('pass_policy.bias')
        with self.assertRaises(ValueError): store_arrays(m, arrays)
        m.blocks[0].message.neighbors[0,0] = 1
        with self.assertRaisesRegex(ValueError,'topology'):
            validate(m)

    def test_roundtrip_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a.pt', Path(folder)/'b.pt'
            m = TorusNet()
            x,y = save_checkpoint(m,a), save_checkpoint(m,b)
            self.assertEqual(x['state_sha256'],y['state_sha256'])
            self.assertEqual(sha256(a),sha256(b))
            restored,_ = restore_checkpoint(a)
            self.assertEqual(state_digest(m),state_digest(restored))
            with self.assertRaises(FileExistsError): save_checkpoint(m,a)
            a.write_bytes(a.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA'): restore_checkpoint(a)

    def test_zero_pass_never_forwards(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(TorusNet,'forward',side_effect=AssertionError('forward')):
            output=Path(folder)/'v0'
            report=zero_pass(output)
            self.assertEqual(report['forward_calls'],0)
            self.assertTrue((output/'genesis.pt').exists())
            with self.assertRaises(FileExistsError): zero_pass(output)

    def test_program_snapshot_repeatability_and_bad_writers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); writer=root/'writer.py'
            writer.write_text("import numpy as np\ndef program(w):\n w['point_policy.weight'][:]=np.arange(120,dtype=np.float32)\n")
            zero_pass(root/'v1',writer)
            self.assertEqual((root/'v1/writer.py').read_bytes(),writer.read_bytes())
            writer.write_text("import numpy as np\ndef program(w):\n w['point_policy.weight'][:]=np.random.random((1,120))\n")
            with self.assertRaisesRegex(ValueError,'reproducible'): zero_pass(root/'bad',writer)
            writer.write_text("def program(w):\n w['point_policy.weight']=w['point_policy.weight'].astype('float64')\n")
            with self.assertRaisesRegex(ValueError,'FP32'): run_writer(writer)


class RulesTests(unittest.TestCase):
    def state(self, black=(), white=(), **kwargs):
        stones=[0]*81
        for i in black: stones[i]=1
        for i in white: stones[i]=2
        return research_state_from_stones(stones,topology=TORUS_9X9,komi=1.5,**kwargs)

    def test_all_81_neighbors_equal_referee(self):
        for i, row in enumerate(torus_neighbors().tolist()):
            self.assertEqual(set(row),set(TORUS_9X9.neighbors(i)))

    def test_wrap_capture_and_liberties(self):
        state=self.state(black=(72,8,9),white=(0,))
        self.assertEqual(liberties_from_board(state,group_from_board(state,0)),{1})
        transition=apply_action(state,1)
        self.assertEqual(transition.captured,(0,))
        self.assertEqual(int(transition.after.stones[0]),0)
        self.assertNotIn(72,legal_actions(state))

    def test_suicide(self):
        state=self.state(white=(72,8,9,1))
        with self.assertRaises(IllegalMoveError) as error: apply_action(state,0)
        self.assertEqual(error.exception.reason.value,'suicide')

    def test_ko_and_superko(self):
        # White at 40 has only liberty 41. Black 41 captures; white 40 would repeat.
        state=self.state(black=(31,39,49),white=(40,32,42,50))
        after=apply_action(state,41).after
        with self.assertRaises(IllegalMoveError) as error: apply_action(after,40)
        self.assertEqual(error.exception.reason.value,'superko')
        # Non-immediate historical board repetition is also illegal.
        empty=initial_state(topology=TORUS_9X9,komi=1.5)
        future=apply_action(empty,0).after.board_key
        historical=self.state(superko_history=(future,empty.board_key))
        with self.assertRaises(IllegalMoveError) as error: apply_action(historical,0)
        self.assertEqual(error.exception.reason.value,'superko')

    def test_pass_terminal_scoring_dame_komi(self):
        state=initial_state(topology=TORUS_9X9,komi=1.5)
        self.assertIn(PASS,legal_actions(state))
        one=apply_action(state,PASS).after
        self.assertEqual(one.superko_history,state.superko_history)
        terminal=apply_action(one,PASS).after
        self.assertTrue(terminal.is_terminal)
        self.assertEqual(legal_actions(terminal),())
        score=score_terminal(terminal)
        self.assertEqual((score.black_area,score.white_area,score.margin_black),(0,0,-1.5))
        with self.assertRaises(IllegalMoveError): apply_action(terminal,PASS)
        # Shared empty region touches both colors: neutral dame, no edge heuristic.
        state=self.state(black=(0,),white=(40,),consecutive_passes=2)
        score=score_terminal(state)
        self.assertEqual((score.black_area,score.white_area),(1,1))
        state=self.state(black=(0,),consecutive_passes=2)
        self.assertEqual(score_terminal(state).black_area,81)


class DiagnosticTests(unittest.TestCase):
    def test_positions_mask_activations_and_wdl(self):
        report=positions(TorusNet(),[{'id':'pass','moves':[0,PASS]}],komi=1.5,activations=True)
        row=report['positions'][0]
        self.assertEqual(row['input'][2],[1.0]*81)
        self.assertEqual(row['input'][3],[1.0]*81)
        self.assertEqual(row['probabilities'][0],0)
        self.assertAlmostEqual(sum(row['probabilities']),1,places=6)
        self.assertEqual(len(row['logits']),82)
        self.assertEqual(len(row['activations']['blocks.7'][0]),81)
        self.assertEqual(wdl_to_side_to_move_utility((1.,0.,0.)),1.)
        self.assertEqual(wdl_to_side_to_move_utility((0.,0.,1.)),-1.)
        self.assertEqual(positions(TorusNet(),[{'moves':[0]}],komi=1.5)['positions'][0]['input'][2],[-1.]*81)

    def test_paired_pass_games_and_watchdog(self):
        m=TorusNet()
        with torch.no_grad(): m.pass_policy.bias.fill_(100)
        report=paired_games(m,komi=1.5,openings=[[],[0,40]],opponent='pass')
        self.assertEqual((report['wins'],report['losses'],report['draws']),(2,2,0))
        self.assertEqual(report['status'],'VALID')
        self.assertEqual([r['candidate_black'] for r in report['games']],[True,False,True,False])
        self.assertEqual(report['games'][2]['opening_moves'],report['games'][3]['opening_moves'])
        limited=paired_games(TorusNet(),komi=1.5,openings=[[]],opponent='pass',watchdog=1)
        self.assertEqual(limited['technical_terminations'],2)
        self.assertEqual(limited['wins']+limited['losses']+limited['draws'],0)
        self.assertEqual(limited['status'],'INVALID')

    def test_different_widths_use_same_mcts(self):
        # Real PUCT on isolated models; use a near-certain PASS to finish cheaply.
        from gocube_golden.torus9_m137_5ch import Torus9M137FiveChannelGraphNet
        ref=Torus9M137FiveChannelGraphNet()
        candidate=TorusNet()
        with torch.no_grad():
            for p in ref.parameters(): p.zero_()
            ref.pass_policy.bias.fill_(100)
            candidate.pass_policy.bias.fill_(100)
        report=paired_games(candidate,komi=1.5,openings=[[]],opponent='reference',reference=ref,
                            simulations=2,tree_reuse=True,watchdog=6)
        self.assertEqual(report['search']['simulations'],2)
        self.assertEqual(report['architectures']['candidate']['width'],120)
        self.assertEqual(report['architectures']['reference']['hidden'],80)
        self.assertGreater(report['inference']['candidate']['calls'],0)
        self.assertGreater(report['inference']['reference']['calls'],0)
        self.assertEqual(report['status'],'VALID')

    def test_budgets_and_failed_round_consumption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); zero_pass(root/'v0')
            positions_path=root/'positions.json'; positions_path.write_text('[{"moves":[]}]')
            for budget in (0,10,25,50):
                session=root/str(budget); init_session(session,budget)
                args=argparse.Namespace(mode='positions',session=session,checkpoint=root/'v0/checkpoint.pt',
                    positions=positions_path,komi=1.5,top_k=5,activations=False)
                if budget==0:
                    with self.assertRaisesRegex(ValueError,'exhausted'): diagnostic(args)
                else:
                    result=diagnostic(args)
                    self.assertEqual(result['round'],1)
                    for i in range(2,budget+1): (session/'rounds'/f'{i:03d}').mkdir()
                    with self.assertRaisesRegex(ValueError,'exhausted'): diagnostic(args)
            session=root/'failed'; init_session(session,10)
            args.session=session; positions_path.write_text('[{"moves":[0,0]}]')
            with self.assertRaises(IllegalMoveError): diagnostic(args)
            self.assertTrue((session/'rounds/001/error.json').exists())

    def test_example_histories_legal(self):
        root=Path(__file__).resolve().parents[1]
        for request in json.loads((root/'positions.json').read_text()):
            replay(request['moves'],1.5)
        for moves in json.loads((root/'openings.json').read_text()):
            self.assertFalse(replay(moves,1.5).is_terminal)


class ReferenceLockTests(unittest.TestCase):
    def test_freeze_rejects_missing_evidence_and_wrong_identity(self):
        from weights import write_json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); checkpoint=root/'ref.pt'; checkpoint.write_bytes(b'isolated fixture')
            meta={'architecture_config':{'input_channels':5,'topology_fingerprint':TORUS_9X9.fingerprint},
                  'observation_shape':[5,81],'komi':1.5,'model_hash':'fixture'}
            write_json(checkpoint.with_suffix('.metadata.json'),meta)
            with self.assertRaisesRegex(ValueError,'Winning'): freeze(checkpoint,[], 'reason')
            evidence={'validity':'VALID','technical_games':0,'valid_games':192,'games':192,'pairs_valid':96,'wins':120,'losses':72,'draws':0,
                      'candidate_model_hash':'fixture','candidate_artifact_sha256':'sha256:'+sha256(checkpoint),
                      'scientific_contract':{'komi':1.5,'input_channels':5,'paired_starts_color_swap':True,
                      'simulations':64,'cpuct':1.25,'fpu':0.,'tree_reuse':True,'watchdog':1000}}
            path=root/'summary.json'; write_json(path,evidence)
            lock=freeze(checkpoint,[path],'explicit reviewed selection')
            self.assertTrue(lock['arena_champion_verified'])
            evidence['pairs_valid']=95; path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'Incomplete'): freeze(checkpoint,[path],'reason')
            evidence['pairs_valid']=96; evidence['candidate_model_hash']='wrong'; path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'identify'): freeze(checkpoint,[path],'reason')


if __name__ == "__main__":
    unittest.main()
