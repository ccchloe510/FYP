"""Metrics and simple analysis helpers."""

from copy import deepcopy
from typing import Dict, List

import numpy as np

try:
    from .init import initialize_params
    from .solver import fit_joint_pg
    from .prox import prox_C
except ImportError:  # pragma: no cover - enables direct script execution
    from init import initialize_params
    from solver import fit_joint_pg
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


def decision_statistics_from_scores(scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Summarize classifier score and margin structure from decision scores."""
    residual = 1.0 - y * scores
    violation = np.maximum(0.0, residual)
    positive_scores = scores[y > 0.0]
    negative_scores = scores[y < 0.0]

    pos_mean = float(np.mean(positive_scores)) if positive_scores.size else float("nan")
    neg_mean = float(np.mean(negative_scores)) if negative_scores.size else float("nan")
    score_gap = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan")

    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "positive_score_mean": pos_mean,
        "negative_score_mean": neg_mean,
        "score_gap": float(score_gap),
        "mean_margin_residual": float(np.mean(residual)),
        "mean_abs_margin_residual": float(np.mean(np.abs(residual))),
        "mean_positive_violation": float(np.mean(violation)),
        "max_positive_violation": float(np.max(violation)),
        "violation_rate": float(np.mean(residual > 0.0)),
        "margin_satisfaction_rate": float(np.mean(residual <= 0.0)),
    }


def decision_statistics(C: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    """Summarize classifier score and margin structure on a split."""
    scores = w @ C + float(b)
    return decision_statistics_from_scores(scores, y)


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
    values = {
        "accuracy": accuracy_from_codes(C, y, params["w"], params["b"]),
        "reconstruction_error": reconstruction_error(X, params["D"], C),
        "code_sparsity": code_sparsity(C),
    }
    values.update(decision_statistics(C, y, params["w"], float(params["b"])))
    return values


def evaluate_joint_model_detailed(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, np.ndarray],
    hyper,
) -> Dict[str, float]:
    """Evaluate a trained joint model and include score/margin diagnostics."""
    return evaluate_joint_model(X, y, params, hyper)


def summarize_joint_result(result: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    params = result["params"]
    values = {
        "accuracy": accuracy_from_codes(params["C"], y, params["w"], params["b"]),
        "reconstruction_error": reconstruction_error(X, params["D"], params["C"]),
        "code_sparsity": code_sparsity(params["C"]),
        "iterations": len(result["history"]["objective"]),
    }
    values.update(decision_statistics(params["C"], y, params["w"], float(params["b"])))
    return values


def joint_component_scale_report(result: Dict) -> Dict[str, Dict[str, float]]:
    """Summarize the scale of joint objective components over optimization."""
    history = result["history"]
    total = np.asarray(history["objective"], dtype=np.float64)
    report = {}
    for key in ("reconstruction", "quadratic_penalty", "hinge_term", "l1_term", "classifier_reg"):
        values = np.asarray(history[key], dtype=np.float64)
        if values.size == 0:
            report[key] = {
                "initial": float("nan"),
                "final": float("nan"),
                "max": float("nan"),
                "final_fraction_of_total": float("nan"),
            }
            continue
        final_fraction = values[-1] / total[-1] if total.size and total[-1] != 0.0 else float("nan")
        report[key] = {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "max": float(values.max()),
            "final_fraction_of_total": float(final_fraction),
        }
    return report


def format_joint_scale_report(scale_report: Dict[str, Dict[str, float]]) -> str:
    lines = ["component | initial | final | max | final/total"]
    for key, stats in scale_report.items():
        lines.append(
            f"{key} | {stats['initial']:.6g} | {stats['final']:.6g} | "
            f"{stats['max']:.6g} | {stats['final_fraction_of_total']:.6g}"
        )
    return "\n".join(lines)


def run_joint_sensitivity_scan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_hyper,
    *,
    rho_values,
    eta_values,
    seeds,
) -> List[Dict[str, float]]:
    """Run a small validation-based scan over rho/eta to diagnose sensitivity."""
    rows = []
    for seed in seeds:
        for rho in rho_values:
            for eta in eta_values:
                hyper = deepcopy(base_hyper)
                hyper.rho = rho
                hyper.eta = eta
                hyper.random_state = seed
                init_params = initialize_params(
                    X_train,
                    y_train,
                    hyper.dictionary_size,
                    seed=seed,
                )
                result = fit_joint_pg(X_train, y_train, hyper, init_params)
                train_metrics = summarize_joint_result(result, X_train, y_train)
                val_metrics = evaluate_joint_model(X_val, y_val, result["params"], hyper)
                scale_report = joint_component_scale_report(result)
                rows.append(
                    {
                        "seed": float(seed),
                        "rho": float(rho),
                        "eta": float(eta),
                        "train_accuracy": float(train_metrics["accuracy"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "train_reconstruction_error": float(train_metrics["reconstruction_error"]),
                        "val_reconstruction_error": float(val_metrics["reconstruction_error"]),
                        "final_reconstruction": float(scale_report["reconstruction"]["final"]),
                        "final_quadratic_penalty": float(scale_report["quadratic_penalty"]["final"]),
                        "final_hinge_term": float(scale_report["hinge_term"]["final"]),
                        "final_l1_term": float(scale_report["l1_term"]["final"]),
                        "status_ok": 1.0 if result["status"] in {"converged", "max_iter_reached"} else 0.0,
                    }
                )
    rows.sort(key=lambda row: (-row["val_accuracy"], row["val_reconstruction_error"]))
    return rows


def summarize_sensitivity_scan(rows: List[Dict[str, float]], top_k: int = 5) -> str:
    """Format the strongest rho/eta diagnostic results as plain text."""
    header = (
        "rank | seed | rho | eta | val_acc | train_acc | "
        "final_recon | final_hinge | final_quad | status_ok"
    )
    lines = [header]
    for rank, row in enumerate(rows[:top_k], start=1):
        lines.append(
            f"{rank} | {int(row['seed'])} | {row['rho']:.6g} | {row['eta']:.6g} | "
            f"{row['val_accuracy']:.6g} | {row['train_accuracy']:.6g} | "
            f"{row['final_reconstruction']:.6g} | {row['final_hinge_term']:.6g} | "
            f"{row['final_quadratic_penalty']:.6g} | {int(row['status_ok'])}"
        )
    return "\n".join(lines)


def format_task_comparison_rows(rows: List[Dict[str, float]]) -> str:
    """Render a compact task comparison table."""
    header = (
        "task | method | train_acc | val_acc | test_acc | "
        "train_recon | val_recon | test_recon | train_sparsity | val_sparsity | test_sparsity"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['task']} | {row['method']} | {row['train_accuracy']:.6g} | {row['val_accuracy']:.6g} | "
            f"{row['test_accuracy']:.6g} | {row['train_reconstruction_error']:.6g} | "
            f"{row['val_reconstruction_error']:.6g} | {row['test_reconstruction_error']:.6g} | "
            f"{row['train_code_sparsity']:.6g} | {row['val_code_sparsity']:.6g} | {row['test_code_sparsity']:.6g}"
        )
    return "\n".join(lines)


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
    print("Decision stats:", decision_statistics(C, y, w, b))
    dummy_result = {
        "params": {"C": C, "D": D, "w": w, "b": np.array(b), "u": np.zeros(C.shape[1])},
        "history": {
            "objective": [10.0, 7.0],
            "reconstruction": [8.0, 5.0],
            "quadratic_penalty": [1.0, 0.8],
            "hinge_term": [0.5, 0.4],
            "l1_term": [0.4, 0.3],
            "classifier_reg": [0.1, 0.1],
        },
    }
    print(format_joint_scale_report(joint_component_scale_report(dummy_result)))


if __name__ == "__main__":
    _self_check()
