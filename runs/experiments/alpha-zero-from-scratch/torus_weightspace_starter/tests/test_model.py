import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import TorusNet, torus_neighbors, write_demo_weights, zero_all_weights, mask_illegal


class TorusExperimentTests(unittest.TestCase):
    def test_torus_wrap(self):
        n = torus_neighbors()
        self.assertEqual(n.shape, (81, 4))
        self.assertEqual(set(n[0].tolist()), {72, 9, 8, 1})
        for i in range(81):
            self.assertEqual(len(set(n[i].tolist())), 4)
            for j in n[i].tolist():
                self.assertIn(i, n[j].tolist())

    def test_zero_genesis(self):
        m = TorusNet(channels=5)
        zero_all_weights(m)
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in m.parameters()))
        with torch.no_grad():
            policy, wdl = m(torch.zeros(2, 5, 81))
        self.assertEqual(tuple(policy.shape), (2,82))
        self.assertEqual(tuple(wdl.shape), (2,3))

    def test_demo_mechanics(self):
        m = TorusNet(channels=5)
        write_demo_weights(m)
        self.assertEqual(m.architecture["parameter_count"], 396009)
        self.assertTrue(any(bool(torch.count_nonzero(p)) for p in m.parameters()))
        obs = torch.zeros(2, 5, 81)
        obs[:, 4, :] = 1
        obs[0, 0, 0] = 1
        obs[0, 4, 0] = 0
        with torch.no_grad():
            p,v,o,s = m.forward_auxiliary(obs)
        self.assertEqual(tuple(p.shape), (2,82))
        self.assertEqual(tuple(v.shape), (2,3))
        self.assertEqual(tuple(o.shape), (2,81,3))
        self.assertEqual(tuple(s.shape), (2,))
        for x in (p,v,o,s): self.assertTrue(torch.isfinite(x).all())
        legal = torch.ones(2,82,dtype=torch.bool)
        legal[0,0] = False
        masked = mask_illegal(p, legal)
        self.assertTrue(torch.isneginf(masked[0,0]))

    def test_torus_translation_equivariance(self):
        m = TorusNet(channels=5)
        write_demo_weights(m)
        obs = torch.zeros(1,5,9,9)
        obs[:,4] = 1
        obs[:,0,0,0] = 1
        obs[:,4,0,0] = 0
        with torch.no_grad():
            p1,_ = m(obs.reshape(1,5,81))
            p2,_ = m(obs.roll((2,3),dims=(-2,-1)).reshape(1,5,81))
        self.assertTrue(torch.allclose(p1[:,:81].reshape(1,9,9).roll((2,3),dims=(-2,-1)),p2[:,:81].reshape(1,9,9),atol=1e-5))
        self.assertTrue(torch.allclose(p1[:,81:],p2[:,81:],atol=1e-5))


if __name__ == "__main__":
    unittest.main()
