"""Proximal operators for the joint model."""

import numpy as np


def prox_C(C_tilde: np.ndarray, step: float, mu: float) -> np.ndarray:
    return np.sign(C_tilde) * np.maximum(np.abs(C_tilde) - step * mu, 0.0)


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
    print("prox_D:", prox_D(np.array([[-1.0, 0.5, 2.0]])))
    print("prox_u:", prox_u(np.array([-1.0, 0.05, 0.5, 2.0]), 0.4, 0.5))


if __name__ == "__main__":
    _self_check()
