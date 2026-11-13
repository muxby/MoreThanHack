"""Viterbi decoding, team-swap symmetry, and noise determinism on synthetic inputs."""

import unittest

import numpy as np

from mtah import features as feat
from mtah import model, noise
from mtah.model import A, D, H, X


def synthetic_game(T=600, seed=0):
    rng = np.random.default_rng(seed)
    P = 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    px = np.empty((T, P), dtype=np.float32)
    py = np.empty((T, P), dtype=np.float32)
    px[:, :11] = rng.uniform(5, 50, 11)
    px[:, 11:] = rng.uniform(55, 100, 11)
    py[:] = rng.uniform(5, 63, P)
    px += rng.normal(0, 0.1, px.shape).astype(np.float32).cumsum(axis=0) * 0.05
    bx = px[:, 3].copy()
    by = py[:, 3].copy()
    period = np.zeros(T, dtype=np.int8)
    period[10:T - 10] = 1
    labels = np.where(period > 0, H, X).astype(np.int8)
    labels[300:400] = D
    return {"name": "synthetic", "px": px, "py": py, "team": team, "bx": bx, "by": by,
            "period": period, "labels": labels, "pitch": (105.0, 68.0)}


class ViterbiTest(unittest.TestCase):
    def test_penalty_removes_flicker(self):
        T = 200
        p = np.tile([0.55, 0.45, 0.0], (T, 1))
        p[1::2] = [0.45, 0.55, 0.0]  # alternate argmax every frame
        period = np.ones(T, dtype=np.int8)
        flicker = model.viterbi(p, period, 0.0, 0.0)
        self.assertGreater(np.sum(np.diff(flicker) != 0), 100)
        smooth = model.viterbi(p, period, 5.0, 5.0)
        self.assertEqual(len(np.unique(smooth)), 1)

    def test_strong_evidence_still_switches(self):
        T = 300
        p = np.tile([0.9, 0.05, 0.05], (T, 1))
        p[150:] = [0.05, 0.9, 0.05]
        out = model.viterbi(p, np.ones(T, dtype=np.int8), 20.0, 20.0)
        self.assertTrue(np.all(out[:150] == H) and np.all(out[150:] == A))

    def test_outside_periods_is_x_and_halves_independent(self):
        p = np.tile([0.9, 0.05, 0.05], (30, 1))
        p[20:] = [0.05, 0.9, 0.05]
        period = np.array([0] * 5 + [1] * 15 + [2] * 10, dtype=np.int8)
        out = model.viterbi(p, period, 1000.0, 1000.0)
        self.assertTrue(np.all(out[:5] == X))
        self.assertTrue(np.all(out[5:20] == H) and np.all(out[20:] == A))


class FeatureSymmetryTest(unittest.TestCase):
    def test_swap_mirrors_contrasts(self):
        g = synthetic_game()
        f1 = feat.base_features(g)
        f2 = feat.base_features(feat.swapped(g))
        rows = g["period"] > 0
        np.testing.assert_allclose(f1["d_h"][rows], f2["d_a"][rows], rtol=1e-5)
        np.testing.assert_allclose(f1["dcx"][rows], -f2["dcx"][rows], rtol=1e-4, atol=1e-4)

    def test_swapped_labels(self):
        g = synthetic_game()
        s = feat.swapped(g)
        self.assertTrue(np.all(s["labels"][g["labels"] == H] == A))
        self.assertTrue(np.all(s["labels"][g["labels"] == D] == D))

    def test_features_nan_outside_play(self):
        g = synthetic_game()
        f = feat.features(g)
        self.assertTrue(f[g["period"] == 0].isna().all().all())


class NoiseTest(unittest.TestCase):
    def test_deterministic(self):
        g = synthetic_game()
        a, b = noise.degrade(g, seed=5), noise.degrade(g, seed=5)
        np.testing.assert_array_equal(a["bx"], b["bx"])
        np.testing.assert_array_equal(a["px"], b["px"])

    def test_hides_ball_during_play(self):
        g = synthetic_game(T=5000)
        live = g["labels"] < 2
        d = noise.degrade(g, seed=1)
        missing = np.isnan(d["bx"])[live].mean()
        self.assertGreater(missing, 0.1)
        self.assertLess(missing, 0.6)


if __name__ == "__main__":
    unittest.main()
