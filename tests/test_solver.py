"""Tests for solver behavior and analytical gradients."""

import unittest

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.config import HyperParams
from src.init import initialize_params
from src.model import gradients, objective
from src.solver import fit_joint_pg


class SolverTests(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        class DummyHyper:
            mu = 0.13
            rho = 0.7
            gamma = 0.2
            eta = 0.9

        rng = np.random.default_rng(0)
        d, m, n = 4, 3, 5
        X = rng.normal(size=(d, n))
        y = rng.choice([-1.0, 1.0], size=n)
        params = {
            "D": rng.uniform(0.05, 0.95, size=(d, m)),
            "C": rng.normal(size=(m, n)),
            "w": rng.normal(size=m),
            "b": np.array(0.17),
            "u": rng.uniform(0.1, 1.2, size=n),
        }
        grads = gradients(params, X, y, DummyHyper())
        eps = 1e-6

        def smooth_total(local_params):
            return objective(local_params, X, y, DummyHyper())["smooth"]

        for key in ("C", "D", "w", "b", "u"):
            arr = np.asarray(params[key])
            idx = tuple(0 for _ in range(arr.ndim))
            plus = {k: v.copy() for k, v in params.items()}
            minus = {k: v.copy() for k, v in params.items()}
            plus[key][idx] += eps
            minus[key][idx] -= eps
            finite_diff = (smooth_total(plus) - smooth_total(minus)) / (2.0 * eps)
            self.assertAlmostEqual(float(np.asarray(grads[key])[idx]), finite_diff, places=6)

    def test_joint_solver_decreases_objective_and_respects_constraints(self):
        rng = np.random.default_rng(0)
        X = np.clip(rng.normal(size=(5, 12)), 0.0, None)
        X = X / max(float(X.max()), 1.0)
        y = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
        hyper = HyperParams(dictionary_size=4, max_iter=10, initial_step=1.0, tol=1e-12)
        init_params = initialize_params(X, y, m=4, seed=0)
        init_obj = objective(init_params, X, y, hyper)["total"]

        result = fit_joint_pg(X, y, hyper, init_params)

        self.assertIn(result["status"], {"converged", "max_iter_reached"})
        self.assertGreater(len(result["history"]["objective"]), 0)
        self.assertLessEqual(result["history"]["objective"][-1], init_obj)
        self.assertTrue(np.all((result["params"]["D"] >= 0.0) & (result["params"]["D"] <= 1.0)))
        self.assertEqual(result["params"]["u"].shape, (12,))
        self.assertGreater(float(np.linalg.norm(result["params"]["w"])), 0.0)

    def test_initialize_u_starts_at_zero_to_activate_coupling(self):
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float64,
        )
        y = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        init_params = initialize_params(X, y, m=2, seed=0)
        np.testing.assert_allclose(init_params["u"], np.zeros_like(y))


if __name__ == "__main__":
    unittest.main()
