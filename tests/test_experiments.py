"""Tests for task suites and multi-task experiment helpers."""

import unittest
from unittest.mock import patch

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import fashion_task_suite, mnist_task_suite, one_vs_rest_suite, report_task_suite, task_catalog
from src.config import default_hyperparams
from src.data import load_task
from src.experiments import (
    benchmark_binary_task,
    fit_svm_on_fixed_dictionary,
    format_method_aggregate_summary,
    format_dictionary_svm_diagnostic,
    format_task_suite_summary,
    summarize_method_aggregate,
)
from src.metrics import (
    code_distribution_summary,
    decision_statistics_from_scores,
    format_code_distribution_report,
    format_overfitting_diagnostic,
    overfitting_diagnostic_summary,
)


class TaskSuiteTests(unittest.TestCase):
    def test_task_catalog_and_suites(self):
        report_tasks = report_task_suite()
        self.assertEqual(len(report_tasks), 4)
        self.assertIn("3 vs 8", task_catalog())
        self.assertIn("fashion::T-shirt vs Shirt", task_catalog())
        self.assertEqual(len(one_vs_rest_suite()), 10)
        self.assertEqual(report_tasks[0].label_description(), "3 vs 8")
        self.assertEqual(mnist_task_suite()[0].dataset, "mnist")
        self.assertEqual(fashion_task_suite()[0].dataset, "fashion_mnist")

    def test_load_task_wrapper(self):
        task = report_task_suite()[0]
        fake_split = (
            np.zeros((2, 3), dtype=np.float64),
            np.array([1.0, -1.0, 1.0]),
            np.zeros((2, 2), dtype=np.float64),
            np.array([1.0, -1.0]),
            np.zeros((2, 1), dtype=np.float64),
            np.array([1.0]),
        )
        with patch("src.data.load_binary_task", return_value=fake_split) as mocked:
            split = load_task(task)
        self.assertEqual(split[0].shape, (2, 3))
        self.assertTrue(mocked.called)
        self.assertEqual(mocked.call_args.kwargs["dataset"], task.dataset)
        self.assertEqual(mocked.call_args.kwargs["positive_labels"], task.positive_labels)

    def test_decision_statistics_from_scores(self):
        scores = np.array([1.5, -0.3, 0.2, -1.0], dtype=np.float64)
        y = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        stats = decision_statistics_from_scores(scores, y)
        self.assertAlmostEqual(stats["violation_rate"], 0.5)
        self.assertAlmostEqual(stats["score_gap"], 1.5)
        self.assertAlmostEqual(stats["margin_satisfaction_rate"], 0.5)

    def test_code_distribution_summary_and_formatter(self):
        C = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        D = np.eye(2, dtype=np.float64)
        X = D @ C
        y = np.array([1.0, -1.0], dtype=np.float64)
        w = np.array([1.0, -1.0], dtype=np.float64)
        stats = code_distribution_summary(C, X, D, y, w, 0.0)
        self.assertAlmostEqual(stats["recon_residual_l2_mean"], 0.0)
        self.assertAlmostEqual(stats["code_sparsity"], 0.5)
        text = format_code_distribution_report({"train": stats, "val": stats, "test": stats})
        self.assertIn("split | code_l2_mean", text)
        self.assertIn("train |", text)

    def test_overfitting_diagnostic_summary_and_formatter(self):
        result = {"history": {"w_norm": [0.5, 6.0]}}
        train = {"accuracy": 1.0, "mean_positive_violation": 0.1, "score_gap": 2.0}
        val = {"accuracy": 0.9, "mean_positive_violation": 0.3, "score_gap": 1.4}
        test = {"accuracy": 0.85, "mean_positive_violation": 0.5, "score_gap": 1.2}
        code_report = {
            "train": {"code_l2_mean": 1.0},
            "val": {"code_l2_mean": 1.1},
            "test": {"code_l2_mean": 1.2},
        }
        summary = overfitting_diagnostic_summary(result, train, val, test, code_report)
        self.assertAlmostEqual(summary["train_test_accuracy_gap"], 0.15)
        self.assertAlmostEqual(summary["test_score_gap_retention"], 0.6)
        text = format_overfitting_diagnostic(summary)
        self.assertIn("metric | value | interpretation", text)
        self.assertIn("diagnosis", text)

    def test_fixed_dictionary_svm_diagnostic(self):
        D = np.eye(2, dtype=np.float64)
        X_train = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        y_train = np.array([1.0, -1.0, 1.0, -1.0])
        X_val = X_train.copy()
        y_val = y_train.copy()
        X_test = X_train.copy()
        y_test = y_train.copy()

        class DummyHyper:
            mu = 0.0
            gamma = 0.1
            initial_step = 1.0
            backtracking_shrink = 0.5
            backtracking_min_step = 1e-8
            max_iter = 5
            tol = 1e-8

        diagnostic = fit_svm_on_fixed_dictionary(
            D, X_train, y_train, X_val, y_val, X_test, y_test, DummyHyper()
        )
        self.assertIn("train_summary", diagnostic)
        self.assertGreaterEqual(diagnostic["test_summary"]["accuracy"], 0.5)
        text = format_dictionary_svm_diagnostic(
            diagnostic["train_summary"],
            diagnostic["val_summary"],
            diagnostic["test_summary"],
            diagnostic,
        )
        self.assertIn("Joint dictionary + separate SVM", text)

    def test_benchmark_includes_separate_prototype_when_joint_uses_simplex(self):
        task = report_task_suite()[0]
        X_train = np.array(
            [[1.0, 0.9, 0.0, 0.1], [0.0, 0.1, 1.0, 0.9]], dtype=np.float64
        )
        y_train = np.array([1.0, 1.0, -1.0, -1.0])
        fake_split = (
            X_train,
            y_train,
            X_train.copy(),
            y_train.copy(),
            X_train.copy(),
            y_train.copy(),
        )

        baseline_hyper = default_hyperparams()
        joint_hyper = default_hyperparams()
        baseline_hyper.dictionary_size = 2
        joint_hyper.dictionary_size = 2
        baseline_hyper.max_iter = 2
        joint_hyper.max_iter = 2
        joint_hyper.code_simplex = True

        with patch("src.experiments.load_task", return_value=fake_split):
            result = benchmark_binary_task(task, baseline_hyper, joint_hyper)

        methods = [row["method"] for row in result["comparison_rows"]]
        self.assertEqual(
            methods,
            ["Raw SVM", "Separate Dict + SVM", "Separate Prototype + SVM", "Joint Prototype + SVM"],
        )
        self.assertIn("separate_prototype", result)

    def test_task_suite_summary_formatter(self):
        rows = [
            {
                "task": "3 vs 8",
                "method": "Raw SVM",
                "train_accuracy": 1.0,
                "val_accuracy": 0.95,
                "test_accuracy": 0.94,
                "train_reconstruction_error": float("nan"),
                "val_reconstruction_error": float("nan"),
                "test_reconstruction_error": float("nan"),
                "train_code_sparsity": float("nan"),
                "val_code_sparsity": float("nan"),
                "test_code_sparsity": float("nan"),
            }
        ]
        summary = format_task_suite_summary([{"comparison_rows": rows}])
        self.assertIn("task | method | train_acc", summary)
        self.assertIn("3 vs 8 | Raw SVM", summary)

    def test_method_aggregate_summary(self):
        rows = [
            {
                "method": "Raw SVM",
                "train_accuracy": 1.0,
                "val_accuracy": 0.95,
                "test_accuracy": 0.94,
                "train_score_gap": 0.1,
                "val_score_gap": 0.2,
                "test_score_gap": 0.15,
                "train_violation_rate": 0.0,
                "val_violation_rate": 0.1,
                "test_violation_rate": 0.2,
                "train_reconstruction_error": float("nan"),
                "val_reconstruction_error": float("nan"),
                "test_reconstruction_error": float("nan"),
                "train_code_sparsity": float("nan"),
                "val_code_sparsity": float("nan"),
                "test_code_sparsity": float("nan"),
                "objective_reconstruction_fraction": float("nan"),
                "objective_quadratic_fraction": float("nan"),
                "objective_hinge_fraction": float("nan"),
            },
            {
                "method": "Raw SVM",
                "train_accuracy": 0.98,
                "val_accuracy": 0.96,
                "test_accuracy": 0.93,
                "train_score_gap": 0.2,
                "val_score_gap": 0.1,
                "test_score_gap": 0.12,
                "train_violation_rate": 0.0,
                "val_violation_rate": 0.2,
                "test_violation_rate": 0.25,
                "train_reconstruction_error": float("nan"),
                "val_reconstruction_error": float("nan"),
                "test_reconstruction_error": float("nan"),
                "train_code_sparsity": float("nan"),
                "val_code_sparsity": float("nan"),
                "test_code_sparsity": float("nan"),
                "objective_reconstruction_fraction": float("nan"),
                "objective_quadratic_fraction": float("nan"),
                "objective_hinge_fraction": float("nan"),
            },
        ]
        summary = summarize_method_aggregate(rows)
        self.assertEqual(summary[0]["method"], "Raw SVM")
        self.assertAlmostEqual(summary[0]["val_accuracy_mean"], 0.955)
        text = format_method_aggregate_summary(summary)
        self.assertIn("method | tasks | train_acc_mean", text)
        self.assertIn("Raw SVM | 2", text)


if __name__ == "__main__":
    unittest.main()
