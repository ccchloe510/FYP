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
from src.data import load_task
from src.experiments import (
    format_method_aggregate_summary,
    format_task_suite_summary,
    summarize_method_aggregate,
)
from src.metrics import decision_statistics_from_scores


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
