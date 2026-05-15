"""Tests for explainability helpers."""

import unittest

import numpy as np

from src.explainability import (
    margin_diagnostics,
    sample_explanation,
    select_representative_indices,
    svm_margin_projection,
)


class ExplainabilityTests(unittest.TestCase):
    def test_margin_diagnostics_separates_correct_violation_and_errors(self):
        C = np.array([[2.0, 0.2, -0.5], [0.0, 0.0, 0.0]])
        y = np.array([1.0, 1.0, 1.0])
        w = np.array([1.0, 0.0])
        b = 0.0

        diagnostics = margin_diagnostics(C, y, w, b)

        np.testing.assert_allclose(diagnostics["signed_margin"], np.array([2.0, 0.2, -0.5]))
        np.testing.assert_array_equal(diagnostics["correct"], np.array([True, True, False]))
        np.testing.assert_array_equal(diagnostics["violated"], np.array([False, True, True]))
        np.testing.assert_array_equal(diagnostics["misclassified"], np.array([False, False, True]))

    def test_select_representative_indices_returns_expected_groups(self):
        C = np.array([[2.0, 0.2, -0.5], [0.0, 0.0, 0.0]])
        y = np.array([1.0, 1.0, 1.0])
        w = np.array([1.0, 0.0])
        diagnostics = margin_diagnostics(C, y, w, 0.0)

        selected = select_representative_indices(diagnostics, max_per_group=2)

        np.testing.assert_array_equal(selected["strong_correct"], np.array([0]))
        np.testing.assert_array_equal(selected["violated_correct"], np.array([1]))
        np.testing.assert_array_equal(selected["misclassified"], np.array([2]))

    def test_sample_explanation_reports_top_atoms_and_contributions(self):
        X = np.array([[1.0], [0.0], [0.5], [0.0]])
        D = np.eye(4, 3)
        C = np.array([[2.0], [0.1], [-1.0]])
        y = np.array([1.0])
        w = np.array([0.5, 4.0, -2.0])

        explanation = sample_explanation(X, D, C, y, w, 0.0, index=0, top_k=2)

        self.assertEqual(int(explanation["index"]), 0)
        np.testing.assert_array_equal(explanation["top_atom_indices"], np.array([0, 2]))
        np.testing.assert_allclose(explanation["top_atom_score_contributions"], np.array([1.0, 2.0]))
        self.assertAlmostEqual(float(explanation["score"]), 3.4)

    def test_svm_margin_projection_uses_geometric_signed_distance(self):
        features = np.array([[2.0, -2.0], [0.0, 0.0]])
        y = np.array([1.0, -1.0])
        w = np.array([2.0, 0.0])
        projection = svm_margin_projection(features, y, w, b=0.0)

        np.testing.assert_allclose(projection["scores"], np.array([4.0, -4.0]))
        np.testing.assert_allclose(projection["x"], np.array([2.0, -2.0]))
        self.assertAlmostEqual(float(projection["margin_distance"]), 0.5)
        self.assertAlmostEqual(float(projection["score_gap"]), 8.0)


if __name__ == "__main__":
    unittest.main()
