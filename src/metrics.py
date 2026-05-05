"""Metrics and simple analysis helpers."""

from typing import Dict

import numpy as np

try:
    from .prox import prox_C
except ImportError:  # pragma: no cover - enables direct script execution
    from prox import prox_C


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


def infer_codes_with_dictionary(
    X: np.ndarray,
    D: np.ndarray,
    mu: float,
    initial_step: float,
    backtracking_shrink: float,
    backtracking_min_step: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Infer sparse codes for new samples with a fixed dictionary."""
    C = np.zeros((D.shape[1], X.shape[1]), dtype=np.float64)
    previous_total = None

    for _ in range(max_iter):
        residual = D @ C - X
        grad_C = D.T @ residual
        current_reconstruction = 0.5 * float(np.sum(residual * residual))
        current_total = current_reconstruction + mu * float(np.sum(np.abs(C)))
        step = initial_step
        accepted = False

        while step >= backtracking_min_step:
            trial_C = prox_C(C - step * grad_C, step, mu)
            diff = trial_C - C
            trial_residual = D @ trial_C - X
            trial_reconstruction = 0.5 * float(np.sum(trial_residual * trial_residual))
            rhs = (
                current_reconstruction
                + float(np.sum(grad_C * diff))
                + float(np.sum(diff * diff)) / (2.0 * step)
            )
            if trial_reconstruction <= rhs:
                accepted = True
                break
            step *= backtracking_shrink

        if not accepted:
            break

        trial_total = trial_reconstruction + mu * float(np.sum(np.abs(trial_C)))
        C = trial_C
        if previous_total is not None:
            rel_change = abs(previous_total - trial_total) / max(1.0, abs(previous_total))
            if rel_change < tol:
                break
        previous_total = trial_total

    return C


def evaluate_joint_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, np.ndarray],
    hyper,
) -> Dict[str, float]:
    """Evaluate a trained joint model on any split by inferring codes with fixed D."""
    C = infer_codes_with_dictionary(
        X=X,
        D=params["D"],
        mu=hyper.mu,
        initial_step=hyper.initial_step,
        backtracking_shrink=hyper.backtracking_shrink,
        backtracking_min_step=hyper.backtracking_min_step,
        max_iter=hyper.max_iter,
        tol=hyper.tol,
    )
    return {
        "accuracy": accuracy_from_codes(C, y, params["w"], params["b"]),
        "reconstruction_error": reconstruction_error(X, params["D"], C),
        "code_sparsity": code_sparsity(C),
    }


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
