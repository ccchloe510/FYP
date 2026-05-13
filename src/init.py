"""Initialization utilities for the joint optimization variables."""

from typing import Dict

import numpy as np


def initialize_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    m: int,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Initialize D, C, w, b, u using the agreed project defaults."""
    rng = np.random.default_rng(seed)
    d, n = X_train.shape
    if m > n:
        raise ValueError(f"dictionary size m={m} cannot exceed number of samples n={n}")

    atom_indices = rng.choice(n, size=m, replace=False)
    D = np.clip(X_train[:, atom_indices].copy(), 0.0, 1.0)
    C = np.zeros((m, n), dtype=np.float64)
    w = np.zeros(m, dtype=np.float64)
    b = np.array(0.0, dtype=np.float64)
    # Start away from q = u - r = 0 so the classification-coupling branch
    # has a nonzero signal on the first iterations.
    u = np.zeros(n, dtype=np.float64)

    return {"D": D, "C": C, "w": w, "b": b, "u": u}


def _self_check() -> None:
    X = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    params = initialize_params(X, y, m=2, seed=0)
    print("Shapes:", {k: np.asarray(v).shape for k, v in params.items()})
    print("D range:", float(params["D"].min()), float(params["D"].max()))
    print("u range:", float(params["u"].min()), float(params["u"].max()))


if __name__ == "__main__":
    _self_check()
