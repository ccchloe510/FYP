"""Tests for baseline implementations."""

import unittest

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import fit_separate_dict_svm, fit_separate_dictionary
from src.config import HyperParams
from src.metrics import summarize_sensitivity_scan


class BaselineTests(unittest.TestCase):
    def test_separate_dictionary_respects_constraints(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0.0, 1.0, size=(6, 16))
        hyper = HyperParams(dictionary_size=4, max_iter=15, initial_step=0.5, tol=1e-12)

        result = fit_separate_dictionary(X, hyper)

        self.assertIn(result["status"], {"converged", "max_iter_reached"})
        self.assertGreater(len(result["history"]["objective"]), 0)
        self.assertTrue(np.all((result["params"]["D"] >= 0.0) & (result["params"]["D"] <= 1.0)))

    def test_separate_baseline_returns_expected_shapes(self):
        rng = np.random.default_rng(2)
        X_pos = rng.normal(loc=0.8, scale=0.05, size=(6, 10))
        X_neg = rng.normal(loc=0.2, scale=0.05, size=(6, 10))
        X = np.clip(np.concatenate([X_pos, X_neg], axis=1), 0.0, 1.0)
        y = np.array([1.0] * 10 + [-1.0] * 10)
        X_train, X_test = X[:, :16], X[:, 16:]
        y_train, y_test = y[:16], y[16:]
        hyper = HyperParams(dictionary_size=4, max_iter=20, initial_step=0.5, tol=1e-12)

        result = fit_separate_dict_svm(X_train, y_train, X_test, y_test, hyper)

        self.assertEqual(result["dictionary"].shape, (6, 4))
        self.assertEqual(result["codes_train"].shape, (4, 16))
        self.assertEqual(result["codes_test"].shape, (4, 4))
        self.assertIn("dictionary_result", result)
        self.assertTrue(np.isfinite(result["metrics"]["train_accuracy"]))
        self.assertTrue(np.isfinite(result["metrics"]["test_accuracy"]))

    def test_sensitivity_summary_formats_rows(self):
        rows = [
            {
                "seed": 7.0,
                "rho": 10.0,
                "eta": 50.0,
                "val_accuracy": 0.94,
                "train_accuracy": 0.96,
                "final_reconstruction": 100.0,
                "final_hinge_term": 8.0,
                "final_quadratic_penalty": 3.0,
                "status_ok": 1.0,
            }
        ]
        summary = summarize_sensitivity_scan(rows, top_k=1)
        self.assertIn("rank | seed | rho | eta", summary)
        self.assertIn("1 | 7 | 10 | 50", summary)


if __name__ == "__main__":
    unittest.main()
