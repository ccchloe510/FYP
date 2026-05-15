"""Proximal operators for the joint model."""

import numpy as np


def project_columns_to_simplex(Z: np.ndarray) -> np.ndarray:
    """Project each column of Z onto {c >= 0, sum(c) = 1}."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("simplex projection expects a 2D matrix")
    sorted_Z = np.sort(Z, axis=0)[::-1]
    cssv = np.cumsum(sorted_Z, axis=0) - 1.0
    rows = np.arange(1, Z.shape[0] + 1, dtype=np.float64)[:, None]
    support = sorted_Z - cssv / rows > 0.0
    rho = np.sum(support, axis=0) - 1
    theta = cssv[rho, np.arange(Z.shape[1])] / (rho + 1.0)
    return np.maximum(Z - theta, 0.0)


def prox_C(C_tilde: np.ndarray, step: float, mu: float, simplex: bool = False) -> np.ndarray:
    if simplex:
        return project_columns_to_simplex(C_tilde)
    return np.sign(C_tilde) * np.maximum(np.abs(C_tilde) - step * mu, 0.0)


def prox_w(w_tilde: np.ndarray, step: float, w_l1: float) -> np.ndarray:
    return np.sign(w_tilde) * np.maximum(np.abs(w_tilde) - step * w_l1, 0.0)


def prox_D(D_tilde: np.ndarray) -> np.ndarray:
    return np.clip(D_tilde, 0.0, 1.0)


def prox_u(u_tilde: np.ndarray, step: float, eta: float) -> np.ndarray:
    tau = step * eta / 2.0
    return np.where(
        u_tilde < 0.0,
        u_tilde,
        np.where(u_tilde <= tau, 0.0, u_tilde - tau),
    )


def _self_check() -> None:
    print("prox_C:", prox_C(np.array([[1.0, -0.5]]), 0.5, 0.1))
    print("prox_C simplex:", prox_C(np.array([[0.8, -0.5], [0.4, 2.0]]), 0.5, 0.1, simplex=True))
    print("prox_w:", prox_w(np.array([1.0, -0.5]), 0.5, 0.1))
    print("prox_D:", prox_D(np.array([[-1.0, 0.5, 2.0]])))
    print("prox_u:", prox_u(np.array([-1.0, 0.05, 0.5, 2.0]), 0.4, 0.5))


if __name__ == "__main__":
    _self_check()
