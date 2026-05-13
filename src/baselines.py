"""Baseline models for comparison with the joint method."""

from typing import Dict

import numpy as np

try:
    from sklearn.svm import LinearSVC
except ImportError as exc:  # pragma: no cover - informative runtime failure
    raise ImportError(
        "scikit-learn is required for src.baselines. Install requirements.txt first."
    ) from exc

try:
    from .prox import prox_C, prox_D
except ImportError:  # pragma: no cover - enables direct script execution
    from prox import prox_C, prox_D


def _copy_array_dict(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: value.copy() for key, value in params.items()}


def _reconstruction_objective(params: Dict[str, np.ndarray], X: np.ndarray, mu: float) -> Dict[str, float]:
    D, C = params["D"], params["C"]
    residual = D @ C - X
    reconstruction = 0.5 * float(np.sum(residual * residual))
    l1_term = mu * float(np.sum(np.abs(C)))
    D_indicator = 0.0 if np.all((D >= 0.0) & (D <= 1.0)) else np.inf
    total = reconstruction + l1_term + D_indicator
    return {
        "total": total,
        "reconstruction": reconstruction,
        "l1_term": l1_term,
        "D_indicator": D_indicator,
    }


def _reconstruction_gradients(params: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
    D, C = params["D"], params["C"]
    residual = D @ C - X
    return {
        "C": D.T @ residual,
        "D": residual @ C.T,
    }


def _separate_prox_step(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    step: float,
    mu: float,
) -> Dict[str, np.ndarray]:
    trial = _copy_array_dict(params)
    trial["C"] = prox_C(params["C"] - step * grads["C"], step, mu)
    trial["D"] = prox_D(params["D"] - step * grads["D"])
    return trial


def _flatten_diff_sq(old: Dict[str, np.ndarray], new: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for key in ("C", "D"):
        diff = np.asarray(new[key]) - np.asarray(old[key])
        total += float(np.sum(diff * diff))
    return total


def _flatten_inner_product(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for key in ("C", "D"):
        total += float(np.sum(np.asarray(left[key]) * np.asarray(right[key])))
    return total


def fit_raw_svm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, hyper) -> Dict:
    model = LinearSVC(C=1.0 / max(hyper.gamma, 1e-8), dual=False, max_iter=5000)
    model.fit(X_train.T, y_train)
    train_acc = float(model.score(X_train.T, y_train))
    test_acc = float(model.score(X_test.T, y_test))
    return {
        "model": model,
        "metrics": {"train_accuracy": train_acc, "test_accuracy": test_acc},
        "params": {"w": model.coef_.ravel(), "b": float(model.intercept_[0])},
    }


def _initialize_dictionary_problem(X_train: np.ndarray, m: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, n = X_train.shape
    if m > n:
        raise ValueError(f"dictionary size m={m} cannot exceed number of samples n={n}")

    atom_indices = rng.choice(n, size=m, replace=False)
    D = np.clip(X_train[:, atom_indices].copy(), 0.0, 1.0)
    C = np.zeros((m, n), dtype=np.float64)
    return {"D": D, "C": C}


def fit_separate_dictionary(
    X_train: np.ndarray,
    hyper,
) -> Dict:
    params = _initialize_dictionary_problem(
        X_train, hyper.dictionary_size, seed=hyper.random_state
    )
    history = {
        "objective": [],
        "reconstruction": [],
        "l1_term": [],
        "step_size": [],
    }
    status = "max_iter_reached"

    for iteration in range(hyper.max_iter):
        current_obj = _reconstruction_objective(params, X_train, hyper.mu)
        grads = _reconstruction_gradients(params, X_train)
        step = hyper.initial_step

        accepted = False
        trial_obj = None
        trial_params = None
        while step >= hyper.backtracking_min_step:
            trial_params = _separate_prox_step(params, grads, step, hyper.mu)
            diff = {
                key: np.asarray(trial_params[key]) - np.asarray(params[key])
                for key in ("C", "D")
            }
            sq_norm = _flatten_diff_sq(params, trial_params)
            rhs = (
                current_obj["reconstruction"]
                + _flatten_inner_product(grads, diff)
                + sq_norm / (2.0 * step)
            )
            trial_obj = _reconstruction_objective(trial_params, X_train, hyper.mu)
            if np.isfinite(trial_obj["total"]) and trial_obj["reconstruction"] <= rhs:
                accepted = True
                break
            step *= hyper.backtracking_shrink

        if not accepted:
            status = "backtracking_failed"
            break

        params = trial_params
        history["objective"].append(trial_obj["total"])
        history["reconstruction"].append(trial_obj["reconstruction"])
        history["l1_term"].append(trial_obj["l1_term"])
        history["step_size"].append(step)

        if iteration > 0:
            prev = history["objective"][-2]
            curr = history["objective"][-1]
            rel_change = abs(prev - curr) / max(1.0, abs(prev))
            if rel_change < hyper.tol:
                status = "converged"
                break

    return {"params": params, "history": history, "status": status}


def _encode_with_fixed_dictionary(
    X: np.ndarray,
    D: np.ndarray,
    mu: float,
    initial_step: float,
    backtracking_shrink: float,
    backtracking_min_step: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    C = np.zeros((D.shape[1], X.shape[1]), dtype=np.float64)

    for iteration in range(max_iter):
        residual = D @ C - X
        step = initial_step
        grad_C = D.T @ residual

        current_reconstruction = 0.5 * float(np.sum(residual * residual))
        current_total = current_reconstruction + mu * float(np.sum(np.abs(C)))
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
        rel_change = abs(current_total - trial_total) / max(1.0, abs(current_total))
        if iteration > 0 and rel_change < tol:
            break

    return C


def fit_separate_dict_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyper,
) -> Dict:
    dict_result = fit_separate_dictionary(X_train, hyper)
    D = dict_result["params"]["D"]
    C_train = dict_result["params"]["C"]
    C_test = _encode_with_fixed_dictionary(
        X_test,
        D,
        hyper.mu,
        hyper.initial_step,
        hyper.backtracking_shrink,
        hyper.backtracking_min_step,
        hyper.max_iter,
        hyper.tol,
    )

    codes_train_rows = C_train.T
    codes_test_rows = C_test.T
    svm = LinearSVC(C=1.0 / max(hyper.gamma, 1e-8), dual=False, max_iter=5000)
    svm.fit(codes_train_rows, y_train)
    train_acc = float(svm.score(codes_train_rows, y_train))
    test_acc = float(svm.score(codes_test_rows, y_test))

    reconstruction = float(np.linalg.norm(X_train - D @ C_train, ord="fro"))
    return {
        "dictionary": D,
        "codes_train": C_train,
        "codes_test": C_test,
        "dictionary_result": dict_result,
        "model": svm,
        "metrics": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "reconstruction_error": reconstruction,
        },
        "params": {"w": svm.coef_.ravel(), "b": float(svm.intercept_[0])},
    }


def _self_check() -> None:
    class DummyHyper:
        dictionary_size = 4
        mu = 0.05
        gamma = 0.1
        initial_step = 0.5
        backtracking_shrink = 0.5
        backtracking_min_step = 1e-8
        max_iter = 10
        tol = 1e-12
        random_state = 7

    rng = np.random.default_rng(2)
    X_pos = rng.normal(loc=0.8, scale=0.05, size=(6, 10))
    X_neg = rng.normal(loc=0.2, scale=0.05, size=(6, 10))
    X = np.clip(np.concatenate([X_pos, X_neg], axis=1), 0.0, 1.0)
    y = np.array([1.0] * 10 + [-1.0] * 10)
    result = fit_separate_dict_svm(X[:, :16], y[:16], X[:, 16:], y[16:], DummyHyper())
    print("Dictionary shape:", result["dictionary"].shape)
    print("Train/Test accuracy:", result["metrics"]["train_accuracy"], result["metrics"]["test_accuracy"])


if __name__ == "__main__":
    _self_check()
