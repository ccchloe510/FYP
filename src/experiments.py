"""Experiment orchestration helpers for multi-task MNIST benchmarking."""

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from sklearn.svm import LinearSVC
except ImportError as exc:  # pragma: no cover - informative runtime failure
    raise ImportError(
        "scikit-learn is required for src.experiments. Install requirements.txt first."
    ) from exc

try:
    from .baselines import fit_separate_dictionary
    from .config import TaskConfig
    from .data import load_mnist_task
    from .init import initialize_params
    from .metrics import (
        code_sparsity,
        decision_statistics_from_scores,
        evaluate_joint_model_detailed,
        infer_codes_with_dictionary,
        joint_component_scale_report,
        reconstruction_error,
        summarize_joint_result,
    )
    from .solver import fit_joint_pg
except ImportError:  # pragma: no cover - enables direct script execution
    from baselines import fit_separate_dictionary
    from config import TaskConfig
    from data import load_mnist_task
    from init import initialize_params
    from metrics import (
        code_sparsity,
        decision_statistics_from_scores,
        evaluate_joint_model_detailed,
        infer_codes_with_dictionary,
        joint_component_scale_report,
        reconstruction_error,
        summarize_joint_result,
    )
    from solver import fit_joint_pg


def _svm_split_summary(model: LinearSVC, X_rows: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(model.decision_function(X_rows), dtype=np.float64).reshape(-1)
    return {
        "accuracy": float(model.score(X_rows, y)),
        **decision_statistics_from_scores(scores, y),
    }


def _code_split_summary(
    model: LinearSVC,
    C: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    D: np.ndarray,
) -> Dict[str, float]:
    scores = np.asarray(model.decision_function(C.T), dtype=np.float64).reshape(-1)
    return {
        "accuracy": float(model.score(C.T, y)),
        "reconstruction_error": reconstruction_error(X, D, C),
        "code_sparsity": code_sparsity(C),
        **decision_statistics_from_scores(scores, y),
    }


def _nan_metrics() -> Dict[str, float]:
    nan = float("nan")
    return {
        "reconstruction_error": nan,
        "code_sparsity": nan,
    }


def _row_from_summaries(
    *,
    task: TaskConfig,
    method: str,
    train_summary: Dict[str, float],
    val_summary: Dict[str, float],
    test_summary: Dict[str, float],
    status: str,
    iterations: float,
    objective_fractions: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    row = {
        "task": task.name,
        "task_description": task.label_description(),
        "method": method,
        "train_accuracy": float(train_summary["accuracy"]),
        "val_accuracy": float(val_summary["accuracy"]),
        "test_accuracy": float(test_summary["accuracy"]),
        "train_reconstruction_error": float(train_summary.get("reconstruction_error", float("nan"))),
        "val_reconstruction_error": float(val_summary.get("reconstruction_error", float("nan"))),
        "test_reconstruction_error": float(test_summary.get("reconstruction_error", float("nan"))),
        "train_code_sparsity": float(train_summary.get("code_sparsity", float("nan"))),
        "val_code_sparsity": float(val_summary.get("code_sparsity", float("nan"))),
        "test_code_sparsity": float(test_summary.get("code_sparsity", float("nan"))),
        "train_score_gap": float(train_summary["score_gap"]),
        "val_score_gap": float(val_summary["score_gap"]),
        "test_score_gap": float(test_summary["score_gap"]),
        "train_violation_rate": float(train_summary["violation_rate"]),
        "val_violation_rate": float(val_summary["violation_rate"]),
        "test_violation_rate": float(test_summary["violation_rate"]),
        "train_mean_margin_residual": float(train_summary["mean_margin_residual"]),
        "val_mean_margin_residual": float(val_summary["mean_margin_residual"]),
        "test_mean_margin_residual": float(test_summary["mean_margin_residual"]),
        "status": status,
        "iterations": float(iterations),
    }
    if objective_fractions is None:
        row.update(
            {
                "objective_reconstruction_fraction": float("nan"),
                "objective_quadratic_fraction": float("nan"),
                "objective_hinge_fraction": float("nan"),
            }
        )
    else:
        row.update(
            {
                "objective_reconstruction_fraction": float(
                    objective_fractions.get("reconstruction", float("nan"))
                ),
                "objective_quadratic_fraction": float(
                    objective_fractions.get("quadratic_penalty", float("nan"))
                ),
                "objective_hinge_fraction": float(
                    objective_fractions.get("hinge_term", float("nan"))
                ),
            }
        )
    return row


def benchmark_binary_task(task: TaskConfig, baseline_hyper, joint_hyper) -> Dict[str, object]:
    """Run raw, separate, and joint models on a single MNIST binary task."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_task(task)

    raw_model = LinearSVC(C=1.0 / max(baseline_hyper.gamma, 1e-8), dual=False, max_iter=5000)
    raw_model.fit(X_train.T, y_train)
    raw_train = _svm_split_summary(raw_model, X_train.T, y_train)
    raw_val = _svm_split_summary(raw_model, X_val.T, y_val)
    raw_test = _svm_split_summary(raw_model, X_test.T, y_test)
    raw_row = _row_from_summaries(
        task=task,
        method="Raw SVM",
        train_summary={**raw_train, **_nan_metrics()},
        val_summary={**raw_val, **_nan_metrics()},
        test_summary={**raw_test, **_nan_metrics()},
        status="trained",
        iterations=float("nan"),
    )

    dict_result = fit_separate_dictionary(X_train, baseline_hyper)
    D = dict_result["params"]["D"]
    C_train = dict_result["params"]["C"]
    C_val = infer_codes_with_dictionary(
        X_val,
        D,
        mu=baseline_hyper.mu,
        initial_step=baseline_hyper.initial_step,
        backtracking_shrink=baseline_hyper.backtracking_shrink,
        backtracking_min_step=baseline_hyper.backtracking_min_step,
        max_iter=baseline_hyper.max_iter,
        tol=baseline_hyper.tol,
    )
    C_test = infer_codes_with_dictionary(
        X_test,
        D,
        mu=baseline_hyper.mu,
        initial_step=baseline_hyper.initial_step,
        backtracking_shrink=baseline_hyper.backtracking_shrink,
        backtracking_min_step=baseline_hyper.backtracking_min_step,
        max_iter=baseline_hyper.max_iter,
        tol=baseline_hyper.tol,
    )

    separate_svm = LinearSVC(C=1.0 / max(baseline_hyper.gamma, 1e-8), dual=False, max_iter=5000)
    separate_svm.fit(C_train.T, y_train)
    separate_train = _code_split_summary(separate_svm, C_train, X_train, y_train, D=D)
    separate_val = _code_split_summary(separate_svm, C_val, X_val, y_val, D=D)
    separate_test = _code_split_summary(separate_svm, C_test, X_test, y_test, D=D)
    separate_row = _row_from_summaries(
        task=task,
        method="Separate Dict + SVM",
        train_summary=separate_train,
        val_summary=separate_val,
        test_summary=separate_test,
        status=dict_result["status"],
        iterations=float(len(dict_result["history"]["objective"])),
    )

    init_params = initialize_params(X_train, y_train, joint_hyper.dictionary_size, seed=joint_hyper.random_state)
    joint_result = fit_joint_pg(X_train, y_train, joint_hyper, init_params)
    joint_train = summarize_joint_result(joint_result, X_train, y_train)
    joint_val = evaluate_joint_model_detailed(X_val, y_val, joint_result["params"], joint_hyper)
    joint_test = evaluate_joint_model_detailed(X_test, y_test, joint_result["params"], joint_hyper)
    scale_report = joint_component_scale_report(joint_result)
    joint_row = _row_from_summaries(
        task=task,
        method="Joint Dict + SVM",
        train_summary=joint_train,
        val_summary=joint_val,
        test_summary=joint_test,
        status=joint_result["status"],
        iterations=float(len(joint_result["history"]["objective"])),
        objective_fractions={
            "reconstruction": scale_report["reconstruction"]["final_fraction_of_total"],
            "quadratic_penalty": scale_report["quadratic_penalty"]["final_fraction_of_total"],
            "hinge_term": scale_report["hinge_term"]["final_fraction_of_total"],
        },
    )

    comparison_rows = [raw_row, separate_row, joint_row]
    return {
        "task": asdict(task),
        "data_shapes": {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape,
        },
        "raw": {
            "model": raw_model,
            "train_summary": raw_train,
            "val_summary": raw_val,
            "test_summary": raw_test,
        },
        "separate": {
            "dictionary_result": dict_result,
            "model": separate_svm,
            "codes_train": C_train,
            "codes_val": C_val,
            "codes_test": C_test,
            "train_summary": separate_train,
            "val_summary": separate_val,
            "test_summary": separate_test,
        },
        "joint": {
            "result": joint_result,
            "train_summary": joint_train,
            "val_summary": joint_val,
            "test_summary": joint_test,
            "scale_report": scale_report,
        },
        "comparison_rows": comparison_rows,
    }


def run_task_suite(tasks: Iterable[TaskConfig], baseline_hyper, joint_hyper) -> List[Dict[str, object]]:
    """Benchmark a suite of tasks and return their results."""
    return [benchmark_binary_task(task, baseline_hyper, joint_hyper) for task in tasks]


def flatten_comparison_rows(task_results: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    """Flatten per-task benchmark results into a single list of method rows."""
    rows: List[Dict[str, float]] = []
    for result in task_results:
        rows.extend(result["comparison_rows"])
    return rows


def format_task_suite_summary(task_results: Iterable[Dict[str, object]]) -> str:
    """Summarize the task suite with the key comparison rows."""
    rows = flatten_comparison_rows(task_results)
    from .metrics import format_task_comparison_rows

    return format_task_comparison_rows(rows)


def summarize_method_aggregate(rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    """Aggregate task-level rows by method."""
    rows = list(rows)
    methods = sorted({row["method"] for row in rows})
    summary_rows = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        numeric_keys = [
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "train_score_gap",
            "val_score_gap",
            "test_score_gap",
            "train_violation_rate",
            "val_violation_rate",
            "test_violation_rate",
            "train_reconstruction_error",
            "val_reconstruction_error",
            "test_reconstruction_error",
            "train_code_sparsity",
            "val_code_sparsity",
            "test_code_sparsity",
            "objective_reconstruction_fraction",
            "objective_quadratic_fraction",
            "objective_hinge_fraction",
        ]
        summary = {"method": method, "tasks": float(len(method_rows))}
        for key in numeric_keys:
            values = np.asarray([row.get(key, np.nan) for row in method_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                summary[f"{key}_mean"] = float("nan")
                summary[f"{key}_std"] = float("nan")
            else:
                summary[f"{key}_mean"] = float(np.mean(finite))
                summary[f"{key}_std"] = float(np.std(finite))
        summary_rows.append(summary)
    return summary_rows


def format_method_aggregate_summary(summary_rows: Iterable[Dict[str, float]]) -> str:
    """Render method-level aggregates as a compact text table."""
    header = (
        "method | tasks | train_acc_mean | val_acc_mean | test_acc_mean | "
        "test_acc_std | val_gap_mean | val_violation_mean | val_recon_mean | "
        "val_sparsity_mean | objective_recon_frac | objective_quad_frac | objective_hinge_frac"
    )
    lines = [header]
    for row in summary_rows:
        lines.append(
            f"{row['method']} | {int(row['tasks'])} | {row['train_accuracy_mean']:.6g} | "
            f"{row['val_accuracy_mean']:.6g} | {row['test_accuracy_mean']:.6g} | "
            f"{row['test_accuracy_std']:.6g} | {row['val_score_gap_mean']:.6g} | "
            f"{row['val_violation_rate_mean']:.6g} | {row['val_reconstruction_error_mean']:.6g} | "
            f"{row['val_code_sparsity_mean']:.6g} | {row['objective_reconstruction_fraction_mean']:.6g} | "
            f"{row['objective_quadratic_fraction_mean']:.6g} | {row['objective_hinge_fraction_mean']:.6g}"
        )
    return "\n".join(lines)


def _self_check() -> None:
    from .config import default_hyperparams, report_task_suite

    baseline_hyper = default_hyperparams()
    joint_hyper = default_hyperparams()
    task = report_task_suite()[0]
    result = benchmark_binary_task(task, baseline_hyper, joint_hyper)
    print("Task:", result["task"]["name"])
    print("Shapes:", result["data_shapes"])
    print(format_task_suite_summary([result]))
    print("Joint scale report:")
    from .metrics import format_joint_scale_report

    print(format_joint_scale_report(result["joint"]["scale_report"]))


if __name__ == "__main__":
    _self_check()
