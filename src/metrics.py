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


def code_sparsity_summary(C: np.ndarray) -> Dict[str, float]:
    """Report exact-zero and practical near-zero sparsity of a code matrix."""
    return {
        "code_sparsity": code_sparsity(C, threshold=1e-10),
        "code_sparsity_1em4": code_sparsity(C, threshold=1e-4),
        "code_sparsity_1em3": code_sparsity(C, threshold=1e-3),
        "code_sparsity_1em2": code_sparsity(C, threshold=1e-2),
    }


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


def code_distribution_summary(
    C: np.ndarray,
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> Dict[str, float]:
    """Summarize code scale, reconstruction residuals, and classifier scores."""
    code_l2 = np.linalg.norm(C, axis=0)
    code_l1 = np.sum(np.abs(C), axis=0)
    residual_l2 = np.linalg.norm(X - D @ C, axis=0)
    values = {
        "code_l2_mean": float(np.mean(code_l2)),
        "code_l2_std": float(np.std(code_l2)),
        "code_l1_mean": float(np.mean(code_l1)),
        "code_l1_std": float(np.std(code_l1)),
        "recon_residual_l2_mean": float(np.mean(residual_l2)),
        "recon_residual_l2_std": float(np.std(residual_l2)),
    }
    values.update(code_sparsity_summary(C))
    values.update(decision_statistics(C, y, w, b))
    return values


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
    }
    values.update(code_sparsity_summary(C))
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
        "iterations": len(result["history"]["objective"]),
    }
    values.update(code_sparsity_summary(params["C"]))
    values.update(decision_statistics(params["C"], y, params["w"], float(params["b"])))
    return values


def joint_code_distribution_report(
    result: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyper,
) -> Dict[str, Dict[str, float]]:
    """Compare optimized train codes with inferred validation/test codes."""
    params = result["params"]
    D, w, b = params["D"], params["w"], float(params["b"])
    C_train = params["C"]
    C_val = infer_codes_with_dictionary(
        X=X_val,
        D=D,
        mu=hyper.mu,
        initial_step=hyper.initial_step,
        backtracking_shrink=hyper.backtracking_shrink,
        backtracking_min_step=hyper.backtracking_min_step,
        max_iter=hyper.max_iter,
        tol=hyper.tol,
    )
    C_test = infer_codes_with_dictionary(
        X=X_test,
        D=D,
        mu=hyper.mu,
        initial_step=hyper.initial_step,
        backtracking_shrink=hyper.backtracking_shrink,
        backtracking_min_step=hyper.backtracking_min_step,
        max_iter=hyper.max_iter,
        tol=hyper.tol,
    )
    return {
        "train": code_distribution_summary(C_train, X_train, D, y_train, w, b),
        "val": code_distribution_summary(C_val, X_val, D, y_val, w, b),
        "test": code_distribution_summary(C_test, X_test, D, y_test, w, b),
    }


def format_code_distribution_report(report: Dict[str, Dict[str, float]]) -> str:
    """Render the train/validation/test code-distribution mismatch report."""
    header = (
        "split | code_l2_mean | code_l2_std | code_l1_mean | recon_residual_l2_mean | "
        "sparsity_1e-3 | score_std | score_gap | violation_rate | mean_pos_violation"
    )
    lines = [header]
    for split in ("train", "val", "test"):
        stats = report[split]
        lines.append(
            f"{split} | {stats['code_l2_mean']:.6g} | {stats['code_l2_std']:.6g} | "
            f"{stats['code_l1_mean']:.6g} | {stats['recon_residual_l2_mean']:.6g} | "
            f"{stats['code_sparsity_1em3']:.6g} | {stats['score_std']:.6g} | "
            f"{stats['score_gap']:.6g} | {stats['violation_rate']:.6g} | "
            f"{stats['mean_positive_violation']:.6g}"
        )
    return "\n".join(lines)


def overfitting_diagnostic_summary(
    result: Dict,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    code_report: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Summarize the main train-to-validation/test overfitting gaps."""
    history = result["history"]
    w_norm = float(history["w_norm"][-1]) if history.get("w_norm") else float("nan")
    train_acc = float(train_metrics["accuracy"])
    val_acc = float(val_metrics["accuracy"])
    test_acc = float(test_metrics["accuracy"])
    train_violation = float(train_metrics["mean_positive_violation"])
    val_violation = float(val_metrics["mean_positive_violation"])
    test_violation = float(test_metrics["mean_positive_violation"])
    train_gap = float(train_metrics["score_gap"])
    val_gap = float(val_metrics["score_gap"])
    test_gap = float(test_metrics["score_gap"])
    train_code_l2 = float(code_report["train"]["code_l2_mean"])
    val_code_l2 = float(code_report["val"]["code_l2_mean"])
    test_code_l2 = float(code_report["test"]["code_l2_mean"])

    return {
        "train_val_accuracy_gap": train_acc - val_acc,
        "train_test_accuracy_gap": train_acc - test_acc,
        "val_test_accuracy_gap": val_acc - test_acc,
        "val_margin_violation_gap": val_violation - train_violation,
        "test_margin_violation_gap": test_violation - train_violation,
        "val_score_gap_retention": val_gap / train_gap if train_gap != 0.0 else float("nan"),
        "test_score_gap_retention": test_gap / train_gap if train_gap != 0.0 else float("nan"),
        "val_code_l2_ratio": val_code_l2 / train_code_l2 if train_code_l2 != 0.0 else float("nan"),
        "test_code_l2_ratio": test_code_l2 / train_code_l2 if train_code_l2 != 0.0 else float("nan"),
        "w_norm": w_norm,
    }


def format_overfitting_diagnostic(summary: Dict[str, float]) -> str:
    """Render overfitting diagnostics and rule-based interpretation."""
    lines = [
        "metric | value | interpretation",
        (
            f"train_val_accuracy_gap | {summary['train_val_accuracy_gap']:.6g} | "
            "large if > 0.05"
        ),
        (
            f"train_test_accuracy_gap | {summary['train_test_accuracy_gap']:.6g} | "
            "large if > 0.08"
        ),
        (
            f"test_margin_violation_gap | {summary['test_margin_violation_gap']:.6g} | "
            "large if test margin is much worse than train"
        ),
        (
            f"test_score_gap_retention | {summary['test_score_gap_retention']:.6g} | "
            "low if < 0.75"
        ),
        (
            f"test_code_l2_ratio | {summary['test_code_l2_ratio']:.6g} | "
            "far from 1 indicates code scale mismatch"
        ),
        (
            f"w_norm | {summary['w_norm']:.6g} | "
            "large values suggest classifier capacity/regularization risk"
        ),
    ]

    recommendations = []
    if summary["train_test_accuracy_gap"] > 0.08 and summary["test_margin_violation_gap"] > 0.25:
        recommendations.append("primary issue: train margin generalizes poorly to test")
    if summary["test_score_gap_retention"] < 0.75:
        recommendations.append("try reducing classifier freedom: higher gamma or earlier stopping")
    if abs(summary["test_code_l2_ratio"] - 1.0) > 0.25:
        recommendations.append("check code inference mismatch before changing classifier parameters")
    if summary["w_norm"] > 5.0:
        recommendations.append("w is large; test stronger classifier regularization or early stopping")
    if not recommendations:
        recommendations.append("no single severe overfitting signal; compare across tasks/seeds")

    lines.append("")
    lines.append("diagnosis")
    lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines)


def joint_component_scale_report(result: Dict) -> Dict[str, Dict[str, float]]:
    """Summarize the scale of joint objective components over optimization."""
    history = result["history"]
    total = np.asarray(history["objective"], dtype=np.float64)
    report = {}
    for key in ("reconstruction", "quadratic_penalty", "hinge_term", "l1_term", "w_l1_term", "classifier_reg"):
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


def format_training_diagnostic_trajectory(result: Dict, points=None) -> str:
    """Render selected iterations from the joint training diagnostic history."""
    history = result["history"]
    n_iters = len(history.get("objective", []))
    if n_iters == 0:
        return "iteration | objective | score_gap | violation_rate | mean_pos_violation | w_norm | mean_u_minus_r | mean_abs_u_minus_r"

    if points is None:
        candidates = [0, 1, 2, 4, 9, n_iters // 2, n_iters - 3, n_iters - 2, n_iters - 1]
        points = sorted({idx for idx in candidates if 0 <= idx < n_iters})

    header = (
        "iteration | objective | score_gap | violation_rate | mean_pos_violation | "
        "w_norm | mean_u_minus_r | mean_abs_u_minus_r | corrected | correction_delta_C"
    )
    lines = [header]
    for idx in points:
        corrected = history.get("code_correction_applied", [0.0] * n_iters)[idx]
        correction_delta = history.get("code_correction_delta_C", [0.0] * n_iters)[idx]
        lines.append(
            f"{idx + 1} | {history['objective'][idx]:.6g} | "
            f"{history['train_score_gap'][idx]:.6g} | {history['train_violation_rate'][idx]:.6g} | "
            f"{history['train_mean_positive_violation'][idx]:.6g} | {history['w_norm'][idx]:.6g} | "
            f"{history['mean_u_minus_r'][idx]:.6g} | {history['mean_abs_u_minus_r'][idx]:.6g} | "
            f"{corrected:.6g} | {correction_delta:.6g}"
        )
    return "\n".join(lines)


def format_inferred_monitor_trajectory(result: Dict, splits=None) -> str:
    """Render fixed-D inferred-code diagnostics collected during joint training."""
    history = result["history"]
    iterations = history.get("monitor_iteration", [])
    if not iterations:
        return (
            "No inferred-code monitor history found. Run fit_joint_pg with "
            "monitor_data={'val': (X_val, y_val)} or similar."
        )

    if splits is None:
        splits = sorted(
            key.replace("monitor_", "").replace("_accuracy", "")
            for key in history
            if key.startswith("monitor_") and key.endswith("_accuracy")
        )

    header_parts = ["iteration"]
    for split in splits:
        header_parts.extend(
            [
                f"{split}_acc",
                f"{split}_gap",
                f"{split}_viol",
                f"{split}_pos_viol",
                f"{split}_code_l2",
            ]
        )
    lines = [" | ".join(header_parts)]

    for row_idx, iteration in enumerate(iterations):
        values = [str(iteration)]
        for split in splits:
            values.extend(
                [
                    f"{history[f'monitor_{split}_accuracy'][row_idx]:.6g}",
                    f"{history[f'monitor_{split}_score_gap'][row_idx]:.6g}",
                    f"{history[f'monitor_{split}_violation_rate'][row_idx]:.6g}",
                    f"{history[f'monitor_{split}_mean_positive_violation'][row_idx]:.6g}",
                    f"{history[f'monitor_{split}_code_l2_mean'][row_idx]:.6g}",
                ]
            )
        lines.append(" | ".join(values))

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
                    code_scale=hyper.init_code_scale,
                    classifier_scale=hyper.init_classifier_scale,
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
        "train_recon | val_recon | test_recon | "
        "val_sparsity_exact | val_sparsity_1e-4 | val_sparsity_1e-3 | val_sparsity_1e-2"
    )
    lines = [header]
    for row in rows:
        val_sparsity = float(row.get("val_code_sparsity", float("nan")))
        val_sparsity_1em4 = float(row.get("val_code_sparsity_1em4", float("nan")))
        val_sparsity_1em3 = float(row.get("val_code_sparsity_1em3", float("nan")))
        val_sparsity_1em2 = float(row.get("val_code_sparsity_1em2", float("nan")))
        lines.append(
            f"{row['task']} | {row['method']} | {row['train_accuracy']:.6g} | {row['val_accuracy']:.6g} | "
            f"{row['test_accuracy']:.6g} | {row['train_reconstruction_error']:.6g} | "
            f"{row['val_reconstruction_error']:.6g} | {row['test_reconstruction_error']:.6g} | "
            f"{val_sparsity:.6g} | {val_sparsity_1em4:.6g} | "
            f"{val_sparsity_1em3:.6g} | {val_sparsity_1em2:.6g}"
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
    print("Sparsity:", code_sparsity_summary(C))
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
