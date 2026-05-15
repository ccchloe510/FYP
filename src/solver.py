"""Joint proximal-gradient solver with backtracking."""

from copy import deepcopy
from typing import Dict

import numpy as np

try:
    from .model import gradients, margin_residual, objective, penalty_residual_q
    from .prox import prox_C, prox_D, prox_u, prox_w
except ImportError:  # pragma: no cover - enables direct script execution
    from model import gradients, margin_residual, objective, penalty_residual_q
    from prox import prox_C, prox_D, prox_u, prox_w


def _copy_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    copied = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = deepcopy(value)
    return copied


def _flatten_diff_sq(old: Dict[str, np.ndarray], new: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for key in ("C", "D", "w", "b", "u"):
        diff = np.asarray(new[key]) - np.asarray(old[key])
        total += float(np.sum(diff * diff))
    return total


def _flatten_inner_product(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for key in ("C", "D", "w", "b", "u"):
        total += float(np.sum(np.asarray(left[key]) * np.asarray(right[key])))
    return total


def gradient_step(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], step: float) -> Dict[str, np.ndarray]:
    trial = _copy_params(params)
    for key in ("C", "D", "w", "b", "u"):
        trial[key] = np.asarray(params[key]) - step * np.asarray(grads[key])
    return trial


def prox_step(trial: Dict[str, np.ndarray], step: float, hyper) -> Dict[str, np.ndarray]:
    proxed = _copy_params(trial)
    proxed["C"] = prox_C(trial["C"], step, hyper.mu)
    proxed["D"] = prox_D(trial["D"])
    proxed["w"] = prox_w(trial["w"], step, getattr(hyper, "w_l1", 0.0))
    proxed["u"] = prox_u(trial["u"], step, hyper.eta)
    proxed["b"] = np.array(float(np.asarray(trial["b"])), dtype=np.float64)
    return proxed


def classification_diagnostics(params: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
    """Track classifier and auxiliary-residual behavior during joint training."""
    C, w, b, u = params["C"], params["w"], float(params["b"]), params["u"]
    scores = w @ C + b
    residual = margin_residual(C, w, b, y)
    q = penalty_residual_q(C, w, b, u, y)
    positive_scores = scores[y > 0.0]
    negative_scores = scores[y < 0.0]
    pos_mean = float(np.mean(positive_scores)) if positive_scores.size else float("nan")
    neg_mean = float(np.mean(negative_scores)) if negative_scores.size else float("nan")
    score_gap = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan")
    return {
        "train_score_gap": float(score_gap),
        "train_violation_rate": float(np.mean(residual > 0.0)),
        "train_mean_positive_violation": float(np.mean(np.maximum(0.0, residual))),
        "w_norm": float(np.linalg.norm(w)),
        "mean_u_minus_r": float(np.mean(q)),
        "mean_abs_u_minus_r": float(np.mean(np.abs(q))),
    }


def _infer_codes_for_monitor(X: np.ndarray, D: np.ndarray, hyper, max_iter: int) -> np.ndarray:
    """Infer codes with fixed D during training diagnostics."""
    C = np.zeros((D.shape[1], X.shape[1]), dtype=np.float64)
    previous_total = None

    for _ in range(max_iter):
        residual = D @ C - X
        grad_C = D.T @ residual
        current_reconstruction = 0.5 * float(np.sum(residual * residual))
        step = hyper.initial_step
        accepted = False

        while step >= hyper.backtracking_min_step:
            trial_C = prox_C(C - step * grad_C, step, hyper.mu)
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
            step *= hyper.backtracking_shrink

        if not accepted:
            break

        trial_total = trial_reconstruction + hyper.mu * float(np.sum(np.abs(trial_C)))
        C = trial_C
        if previous_total is not None:
            rel_change = abs(previous_total - trial_total) / max(1.0, abs(previous_total))
            if rel_change < hyper.tol:
                break
        previous_total = trial_total

    return C


def _inferred_code_diagnostics(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    hyper,
    max_iter: int,
) -> Dict[str, float]:
    """Evaluate current w,b on codes inferred from current D."""
    C = _infer_codes_for_monitor(X, params["D"], hyper, max_iter)
    w, b = params["w"], float(params["b"])
    scores = w @ C + b
    residual = margin_residual(C, w, b, y)
    positive_scores = scores[y > 0.0]
    negative_scores = scores[y < 0.0]
    pos_mean = float(np.mean(positive_scores)) if positive_scores.size else float("nan")
    neg_mean = float(np.mean(negative_scores)) if negative_scores.size else float("nan")
    score_gap = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan")
    return {
        "accuracy": float(np.mean(np.where(scores >= 0.0, 1.0, -1.0) == y)),
        "score_gap": float(score_gap),
        "violation_rate": float(np.mean(residual > 0.0)),
        "mean_positive_violation": float(np.mean(np.maximum(0.0, residual))),
        "code_l2_mean": float(np.mean(np.linalg.norm(C, axis=0))),
    }


def _initialize_monitor_history(history: Dict[str, list], monitor_data) -> None:
    if not monitor_data:
        return
    history["monitor_iteration"] = []
    for split in monitor_data:
        for metric in ("accuracy", "score_gap", "violation_rate", "mean_positive_violation", "code_l2_mean"):
            history[f"monitor_{split}_{metric}"] = []


def _append_monitor_history(
    history: Dict[str, list],
    params: Dict[str, np.ndarray],
    hyper,
    monitor_data,
    iteration: int,
    max_iter: int,
) -> None:
    if not monitor_data:
        return
    history["monitor_iteration"].append(iteration + 1)
    for split, (X_split, y_split) in monitor_data.items():
        diagnostics = _inferred_code_diagnostics(params, X_split, y_split, hyper, max_iter)
        for metric, value in diagnostics.items():
            history[f"monitor_{split}_{metric}"].append(value)


def _apply_code_correction(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    hyper,
) -> Dict[str, np.ndarray]:
    """Move optimized training C toward fixed-D inferred C to reduce deployment mismatch."""
    corrected = _copy_params(params)
    max_iter = int(getattr(hyper, "code_correction_max_iter", 50))
    blend = float(getattr(hyper, "code_correction_blend", 1.0))
    blend = min(1.0, max(0.0, blend))
    inferred_C = _infer_codes_for_monitor(X, corrected["D"], hyper, max_iter)
    corrected_C = (1.0 - blend) * corrected["C"] + blend * inferred_C
    corrected["C"] = corrected_C
    if bool(getattr(hyper, "code_correction_update_u", True)):
        corrected["u"] = margin_residual(corrected_C, corrected["w"], float(corrected["b"]), y)
    return corrected


def fit_joint_pg(
    X: np.ndarray,
    y: np.ndarray,
    hyper,
    init_params: Dict[str, np.ndarray],
    monitor_data=None,
    monitor_every: int = 10,
    monitor_code_max_iter: int = 50,
) -> Dict:
    params = _copy_params(init_params)
    history = {
        "objective": [],
        "smooth": [],
        "nonsmooth": [],
        "reconstruction": [],
        "classifier_reg": [],
        "quadratic_penalty": [],
        "hinge_term": [],
        "l1_term": [],
        "w_l1_term": [],
        "step_size": [],
        "train_score_gap": [],
        "train_violation_rate": [],
        "train_mean_positive_violation": [],
        "w_norm": [],
        "mean_u_minus_r": [],
        "mean_abs_u_minus_r": [],
        "code_correction_applied": [],
        "code_correction_delta_C": [],
    }
    _initialize_monitor_history(history, monitor_data)
    status = "max_iter_reached"

    for iteration in range(hyper.max_iter):
        current_obj = objective(params, X, y, hyper)
        grads = gradients(params, X, y, hyper)
        step = hyper.initial_step

        accepted = False
        trial_obj = None
        trial_params = None
        while step >= hyper.backtracking_min_step:
            grad_trial = gradient_step(params, grads, step)
            trial_params = prox_step(grad_trial, step, hyper)
            trial_obj = objective(trial_params, X, y, hyper)
            diff = {
                key: np.asarray(trial_params[key]) - np.asarray(params[key])
                for key in ("C", "D", "w", "b", "u")
            }
            sq_norm = _flatten_diff_sq(params, trial_params)
            rhs = (
                current_obj["smooth"]
                + _flatten_inner_product(grads, diff)
                + sq_norm / (2.0 * step)
            )
            if np.isfinite(trial_obj["total"]) and trial_obj["smooth"] <= rhs:
                accepted = True
                break
            step *= hyper.backtracking_shrink

        if not accepted:
            status = "backtracking_failed"
            break

        correction_every = int(getattr(hyper, "code_correction_every", 0))
        should_correct = correction_every > 0 and (iteration + 1) % correction_every == 0
        correction_delta_C = 0.0
        if should_correct:
            corrected_params = _apply_code_correction(trial_params, X, y, hyper)
            correction_delta_C = float(np.linalg.norm(corrected_params["C"] - trial_params["C"]))
            trial_params = corrected_params
            trial_obj = objective(trial_params, X, y, hyper)

        params = trial_params
        history["objective"].append(trial_obj["total"])
        history["smooth"].append(trial_obj["smooth"])
        history["nonsmooth"].append(trial_obj["nonsmooth"])
        history["reconstruction"].append(trial_obj["reconstruction"])
        history["classifier_reg"].append(trial_obj["classifier_reg"])
        history["quadratic_penalty"].append(trial_obj["quadratic_penalty"])
        history["hinge_term"].append(trial_obj["hinge_term"])
        history["l1_term"].append(trial_obj["l1_term"])
        history["w_l1_term"].append(trial_obj.get("w_l1_term", 0.0))
        history["step_size"].append(step)
        history["code_correction_applied"].append(float(should_correct))
        history["code_correction_delta_C"].append(correction_delta_C)
        diagnostics = classification_diagnostics(params, y)
        for key, value in diagnostics.items():
            history[key].append(value)
        if monitor_data and ((iteration + 1) == 1 or (iteration + 1) % monitor_every == 0):
            _append_monitor_history(
                history,
                params,
                hyper,
                monitor_data,
                iteration,
                monitor_code_max_iter,
            )

        if iteration > 0:
            prev = history["objective"][-2]
            curr = history["objective"][-1]
            rel_change = abs(prev - curr) / max(1.0, abs(prev))
            if rel_change < hyper.tol:
                status = "converged"
                break

    return {"params": params, "history": history, "status": status}


def _self_check() -> None:
    class DummyHyper:
        mu = 0.05
        rho = 1.0
        gamma = 0.1
        w_l1 = 0.0
        eta = 1.0
        initial_step = 1.0
        backtracking_shrink = 0.5
        backtracking_min_step = 1e-8
        max_iter = 5
        tol = 1e-12

    rng = np.random.default_rng(0)
    X = np.clip(rng.normal(size=(5, 12)), 0.0, None)
    X = X / max(float(X.max()), 1.0)
    y = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    params = {
        "D": rng.uniform(0.0, 1.0, size=(5, 4)),
        "C": np.zeros((4, 12), dtype=np.float64),
        "w": np.zeros(4, dtype=np.float64),
        "b": np.array(0.0, dtype=np.float64),
        "u": np.ones(12, dtype=np.float64),
    }
    result = fit_joint_pg(X, y, DummyHyper(), params)
    print("Status:", result["status"])
    print("Objective history:", result["history"]["objective"])


if __name__ == "__main__":
    _self_check()
