"""Metrics and simple analysis helpers."""

from typing import Dict

import numpy as np


def predict_from_codes(C: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    scores = w @ C + float(b)
    return np.where(scores >= 0.0, 1.0, -1.0)


def accuracy_from_codes(C: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    preds = predict_from_codes(C, w, b)
    return float(np.mean(preds == y))


def reconstruction_error(X: np.ndarray, D: np.ndarray, C: np.ndarray) -> float:
    return float(np.linalg.norm(X - D @ C, ord="fro"))


def code_sparsity(C: np.ndarray, threshold: float = 1e-10) -> float:
    return float(np.mean(np.abs(C) <= threshold))


def summarize_joint_result(result: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    params = result["params"]
    return {
        "accuracy": accuracy_from_codes(params["C"], y, params["w"], params["b"]),
        "reconstruction_error": reconstruction_error(X, params["D"], params["C"]),
        "code_sparsity": code_sparsity(params["C"]),
        "iterations": len(result["history"]["objective"]),
    }


def _self_check() -> None:
    C = np.array([[1.0, -1.0, 0.0], [0.5, -0.2, 0.1]])
    w = np.array([0.3, -0.1])
    b = 0.05
    y = np.array([1.0, -1.0, 1.0])
    D = np.array([[0.2, 0.7], [0.6, 0.1]])
    X = D @ C
    print("Predictions:", predict_from_codes(C, w, b))
    print("Accuracy:", accuracy_from_codes(C, y, w, b))
    print("Reconstruction error:", reconstruction_error(X, D, C))
    print("Code sparsity:", code_sparsity(C))


if __name__ == "__main__":
    _self_check()
