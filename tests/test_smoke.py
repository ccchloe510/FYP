"""Smoke tests for core math helpers."""

import unittest

import numpy as np

from src.model import gradients, objective
from src.prox import prox_C, prox_D, prox_u


class DummyHyper:
    mu = 0.1
    rho = 1.0
    gamma = 0.2
    eta = 0.5


class SmokeTests(unittest.TestCase):
    def test_prox_constraints(self):
        np.testing.assert_allclose(
            prox_C(np.array([[1.0, -0.5]]), 0.5, 0.1),
            np.array([[0.95, -0.45]]),
        )
        np.testing.assert_allclose(
            prox_D(np.array([[-1.0, 0.5, 2.0]])),
            np.array([[0.0, 0.5, 1.0]]),
        )
        np.testing.assert_allclose(
            prox_u(np.array([-1.0, 0.05, 0.5, 2.0]), 0.4, 0.5),
            np.array([-1.0, 0.0, 0.4, 1.9]),
        )

    def test_objective_finite(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        params = {
            "D": np.array([[0.1, 0.0], [0.0, 0.2]]),
            "C": np.array([[0.5, 0.1], [0.0, 0.2]]),
            "w": np.array([0.2, -0.1]),
            "b": np.array(0.0),
            "u": np.array([0.5, 0.2]),
        }
        y = np.array([1.0, -1.0])
        values = objective(params, X, y, DummyHyper())
        grads = gradients(params, X, y, DummyHyper())

        self.assertTrue(np.isfinite(values["total"]))
        self.assertEqual(grads["C"].shape, params["C"].shape)
        self.assertEqual(grads["D"].shape, params["D"].shape)
        self.assertEqual(grads["w"].shape, params["w"].shape)


if __name__ == "__main__":
    unittest.main()
