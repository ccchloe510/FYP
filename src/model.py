"""Objective and gradient definitions for the joint optimization model.

The auxiliary variable ``u`` is an unconstrained copy of the margin residual.
It is coupled to the classifier through a quadratic penalty and receives the
hinge penalty directly.
"""

from typing import Dict

import numpy as np


def margin_residual(C: np.ndarray, w: np.ndarray, b: float, y: np.ndarray) -> np.ndarray:
    return 1.0 - y * (w @ C + b)


def penalty_residual_q(
    C: np.ndarray, w: np.ndarray, b: float, u: np.ndarray, y: np.ndarray
) -> np.ndarray:
    return u - margin_residual(C, w, b, y)


def smooth_objective(params: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, hyper) -> Dict[str, float]:
    D, C, w, b, u = params["D"], params["C"], params["w"], float(params["b"]), params["u"]
    E = D @ C - X
    q = penalty_residual_q(C, w, b, u, y)

    reconstruction = 0.5 * float(np.sum(E * E))
    classifier_reg = 0.5 * hyper.gamma * float(np.dot(w, w))
    quadratic_penalty = 0.5 * hyper.rho * float(np.dot(q, q))
    smooth = reconstruction + classifier_reg + quadratic_penalty
    return {
        "smooth": smooth,
        "reconstruction": reconstruction,
        "classifier_reg": classifier_reg,
        "quadratic_penalty": quadratic_penalty,
    }


def nonsmooth_objective(params: Dict[str, np.ndarray], hyper) -> Dict[str, float]:
    C, D, u = params["C"], params["D"], params["u"]
    l1_term = hyper.mu * float(np.sum(np.abs(C)))
    hinge_term = 0.5 * hyper.eta * float(np.sum(np.maximum(0.0, u)))
    D_indicator = 0.0 if np.all((D >= 0.0) & (D <= 1.0)) else np.inf
    nonsmooth = l1_term + hinge_term + D_indicator
    return {
        "nonsmooth": nonsmooth,
        "l1_term": l1_term,
        "hinge_term": hinge_term,
        "D_indicator": D_indicator,
    }


def objective(params: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, hyper) -> Dict[str, float]:
    smooth = smooth_objective(params, X, y, hyper)
    nonsmooth = nonsmooth_objective(params, hyper)
    total = smooth["smooth"] + nonsmooth["nonsmooth"]
    return {"total": total, **smooth, **nonsmooth}


def gradients(params: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, hyper) -> Dict[str, np.ndarray]:
    D, C, w, b, u = params["D"], params["C"], params["w"], float(params["b"]), params["u"]
    E = D @ C - X
    q = penalty_residual_q(C, w, b, u, y)
    s = q * y

    grad_C = D.T @ E + hyper.rho * np.outer(w, s)
    grad_D = E @ C.T
    grad_w = hyper.gamma * w + hyper.rho * (C @ s)
    grad_b = np.array(hyper.rho * np.sum(s), dtype=np.float64)
    grad_u = hyper.rho * q
    return {"C": grad_C, "D": grad_D, "w": grad_w, "b": grad_b, "u": grad_u}


def _self_check() -> None:
    class DummyHyper:
        mu = 0.1
        rho = 1.0
        gamma = 0.2
        eta = 0.5

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
    print("Objective:", values)
    print("Gradient shapes:", {k: np.asarray(v).shape for k, v in grads.items()})


if __name__ == "__main__":
    _self_check()
