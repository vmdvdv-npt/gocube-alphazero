"""Isolated harness tests. Fixtures never load a production game or model."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import session as s
from summarize import summarize


class PassModel(torch.nn.Module):
    def forward(self,x):
        p=torch.zeros((len(x),82),device=x.device)
        p[:,81]=100
        return p,torch.zeros((len(x),3),device=x.device)


class SessionTests(unittest.TestCase):
    def make_session(self,path,budget=2):
        path.mkdir()
        (path/'rounds').mkdir()
        s.write_json(path/'session.json',{'schema':'weightspace-session-v002','budget':budget,
                     'allocation':{'diagnostic':budget,'development':budget,'holdout':48}})

    def test_atomic_budget_and_failed_round_consumed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(s,'sources',return_value={}):
            path=Path(tmp)/'session';self.make_session(path)
            def attempt(i):
                try:
                    with s.reserve(path,'diagnostic',{'test':i}):
                        if i==0:raise RuntimeError('fixture failure')
                    return True
                except (ValueError,RuntimeError):return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(attempt,range(8)))
            self.assertEqual(len(list((path/'rounds').iterdir())),2)
            self.assertTrue((path/'rounds/001/error.json').exists())

    def test_freeze_closes_development(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(s,'sources',return_value={}):
            path=Path(tmp)/'session';self.make_session(path)
            s.write_json(path/'freeze.json',{})
            with self.assertRaisesRegex(ValueError,'closed'):
                with s.reserve(path,'development',{}):pass

    def test_policy_game_color_pair_and_same_state_wdl(self):
        result=s.arena(PassModel(),PassModel(),[[]],simulations=0,watchdog=4)
        self.assertEqual((result['wins'],result['losses']),(1,1))
        self.assertEqual(result['status'],'VALID')
        for game in result['games']:
            self.assertEqual(len(game['trace']),2)
            self.assertEqual(game['trace'][0]['comparison_wdl'][0],game['trace'][0]['comparison_wdl'][1])

    def test_watchdog_never_becomes_draw(self):
        result=s.arena(PassModel(),PassModel(),[[]],simulations=0,watchdog=1)
        self.assertEqual(result['status'],'INVALID')
        self.assertEqual(result['draws'],0)
        self.assertEqual(result['technical_terminations'],2)

    def test_analysis_keeps_pairs_and_incomplete_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);folder=path/'rounds/001';folder.mkdir(parents=True)
            s.write_json(folder/'request.json',{'phase':'holdout','comparison':'v002-v001-policy','chunk':0})
            s.write_json(folder/'report.json',s.arena(PassModel(),PassModel(),[[]],simulations=0))
            r=summarize(path)['comparisons']['v002-v001-policy']
            self.assertEqual(r['pairs'],1)
            self.assertEqual(r['score_rate'],.5)
            self.assertEqual(r['status'],'INCOMPLETE_OR_INVALID')
            self.assertEqual(r['same_states_wdl_brier_game_mean'][0],r['same_states_wdl_brier_game_mean'][1])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
