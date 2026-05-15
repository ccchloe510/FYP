"""Explainability helpers for margin, reconstruction, and atom contributions."""

from typing import Dict, Iterable, Optional

import numpy as np


def margin_diagnostics(C: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, np.ndarray]:
    """Return per-sample score, margin, residual, correctness, and violation arrays."""
    scores = np.asarray(w) @ np.asarray(C) + float(b)
    y = np.asarray(y, dtype=np.float64)
    signed_margin = y * scores
    residual = 1.0 - signed_margin
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    return {
        "scores": scores,
        "signed_margin": signed_margin,
        "residual": residual,
        "positive_violation": np.maximum(0.0, residual),
        "predictions": predictions,
        "correct": predictions == y,
        "violated": residual > 0.0,
        "misclassified": predictions != y,
    }


def select_representative_indices(
    diagnostics: Dict[str, np.ndarray],
    *,
    max_per_group: int = 3,
) -> Dict[str, np.ndarray]:
    """Select representative correct, margin-violating, and misclassified samples."""
    residual = np.asarray(diagnostics["residual"])
    signed_margin = np.asarray(diagnostics["signed_margin"])
    correct = np.asarray(diagnostics["correct"], dtype=bool)
    violated = np.asarray(diagnostics["violated"], dtype=bool)
    misclassified = np.asarray(diagnostics["misclassified"], dtype=bool)

    strong_correct = np.where(correct & ~violated)[0]
    violated_correct = np.where(correct & violated)[0]
    wrong = np.where(misclassified)[0]

    strong_order = strong_correct[np.argsort(signed_margin[strong_correct])[::-1]]
    violated_order = violated_correct[np.argsort(residual[violated_correct])[::-1]]
    wrong_order = wrong[np.argsort(residual[wrong])[::-1]]

    return {
        "strong_correct": strong_order[:max_per_group],
        "violated_correct": violated_order[:max_per_group],
        "misclassified": wrong_order[:max_per_group],
    }


def sample_explanation(
    X: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    index: int,
    *,
    top_k: int = 6,
) -> Dict[str, np.ndarray]:
    """Explain one sample through reconstruction, active atoms, and score contributions."""
    diagnostics = margin_diagnostics(C, y, w, b)
    x = X[:, index]
    c = C[:, index]
    reconstruction = D @ c
    residual_image = x - reconstruction
    atom_order = np.argsort(np.abs(c))[::-1][:top_k]
    score_contributions = w * c
    contribution_order = np.argsort(np.abs(score_contributions))[::-1][:top_k]

    return {
        "index": int(index),
        "x": x,
        "code": c,
        "reconstruction": reconstruction,
        "residual_image": residual_image,
        "top_atom_indices": atom_order,
        "top_atom_coefficients": c[atom_order],
        "top_atom_weights": w[atom_order],
        "top_atom_score_contributions": score_contributions[atom_order],
        "top_contribution_indices": contribution_order,
        "top_contribution_values": score_contributions[contribution_order],
        "score": np.array(diagnostics["scores"][index]),
        "signed_margin": np.array(diagnostics["signed_margin"][index]),
        "margin_residual": np.array(diagnostics["residual"][index]),
        "positive_violation": np.array(diagnostics["positive_violation"][index]),
        "prediction": np.array(diagnostics["predictions"][index]),
        "label": np.array(y[index]),
        "correct": np.array(diagnostics["correct"][index]),
        "violated": np.array(diagnostics["violated"][index]),
    }


def format_sample_explanation(explanation: Dict[str, np.ndarray]) -> str:
    """Render a compact text explanation for one sample."""
    lines = [
        "field | value",
        f"index | {int(explanation['index'])}",
        f"label | {float(explanation['label']):.6g}",
        f"prediction | {float(explanation['prediction']):.6g}",
        f"correct | {bool(explanation['correct'])}",
        f"score | {float(explanation['score']):.6g}",
        f"signed_margin | {float(explanation['signed_margin']):.6g}",
        f"margin_residual | {float(explanation['margin_residual']):.6g}",
        f"positive_violation | {float(explanation['positive_violation']):.6g}",
        f"violated | {bool(explanation['violated'])}",
        "",
        "atom | coeff | classifier_weight | score_contribution",
    ]
    for atom, coeff, weight, contribution in zip(
        explanation["top_atom_indices"],
        explanation["top_atom_coefficients"],
        explanation["top_atom_weights"],
        explanation["top_atom_score_contributions"],
    ):
        lines.append(f"{int(atom)} | {coeff:.6g} | {weight:.6g} | {contribution:.6g}")
    return "\n".join(lines)


def plot_margin_distribution(diagnostics: Dict[str, np.ndarray], ax=None, title: str = "Margin distribution"):
    """Plot signed margins and mark the decision boundary and unit margin."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    signed_margin = np.asarray(diagnostics["signed_margin"])
    ax.hist(signed_margin, bins=30, alpha=0.8, color="#4c78a8")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="decision boundary")
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="unit margin")
    ax.set_title(title)
    ax.set_xlabel("signed margin y * score")
    ax.set_ylabel("count")
    ax.legend()
    return ax


def svm_margin_projection(features: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, np.ndarray]:
    """Project high-dimensional features into a 2D SVM-margin view.

    The x-axis is geometric signed distance to the decision boundary:

        (w^T x + b) / ||w||

    The y-axis is the leading PCA direction after removing the classifier
    direction. This gives a standard SVM-style visualization even when the
    original feature space is high-dimensional.
    """
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = float(b)
    w_norm = float(np.linalg.norm(w))
    if w_norm == 0.0:
        raise ValueError("Cannot build SVM margin projection with zero classifier weight.")

    scores = w @ features + b
    x_coord = scores / w_norm
    w_unit = w / w_norm
    centered = features - np.mean(features, axis=1, keepdims=True)
    orthogonal = centered - np.outer(w_unit, w_unit @ centered)

    if orthogonal.shape[1] <= 1 or np.allclose(orthogonal, 0.0):
        y_coord = np.zeros(features.shape[1], dtype=np.float64)
    else:
        _, _, vt = np.linalg.svd(orthogonal, full_matrices=False)
        y_coord = vt[0]

    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    return {
        "x": x_coord,
        "y": y_coord,
        "scores": scores,
        "labels": y,
        "predictions": predictions,
        "correct": predictions == y,
        "margin_distance": np.array(1.0 / w_norm),
        "score_gap": np.array(
            float(np.mean(scores[y > 0.0]) - np.mean(scores[y < 0.0]))
            if np.any(y > 0.0) and np.any(y < 0.0)
            else float("nan")
        ),
    }


def plot_svm_margin_projection(
    projection: Dict[str, np.ndarray],
    ax=None,
    title: str = "SVM margin projection",
):
    """Plot a 2D SVM-style margin view with boundary and two margin lines."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    x = np.asarray(projection["x"])
    y_coord = np.asarray(projection["y"])
    labels = np.asarray(projection["labels"])
    correct = np.asarray(projection["correct"], dtype=bool)
    margin = float(projection["margin_distance"])

    for label, color, marker in ((1.0, "#1f77b4", "o"), (-1.0, "#ff7f0e", "s")):
        mask = labels == label
        ax.scatter(
            x[mask & correct],
            y_coord[mask & correct],
            c=color,
            marker=marker,
            alpha=0.65,
            s=28,
            label=f"y={label:.0f} correct",
        )
        ax.scatter(
            x[mask & ~correct],
            y_coord[mask & ~correct],
            facecolors="none",
            edgecolors=color,
            marker=marker,
            linewidths=1.4,
            s=52,
            label=f"y={label:.0f} wrong",
        )

    ax.axvline(0.0, color="black", linewidth=1.4, label="decision boundary")
    ax.axvline(margin, color="#d62728", linestyle="--", linewidth=1.2, label="+ margin")
    ax.axvline(-margin, color="#d62728", linestyle="--", linewidth=1.2, label="- margin")
    ax.set_title(title)
    ax.set_xlabel("signed distance to boundary")
    ax.set_ylabel("orthogonal variation")
    ax.grid(alpha=0.2)
    return ax


def plot_sample_explanation(
    explanation: Dict[str, np.ndarray],
    D: np.ndarray,
    *,
    image_shape=(28, 28),
    top_k: int = 6,
):
    """Plot original image, reconstruction, residual, top atoms, and score contributions."""
    import matplotlib.pyplot as plt

    top_atoms = explanation["top_atom_indices"][:top_k]
    coeffs = explanation["top_atom_coefficients"][:top_k]
    contributions = explanation["top_atom_score_contributions"][:top_k]

    fig = plt.figure(figsize=(3 * max(top_k, 3), 7))
    grid = fig.add_gridspec(2, max(top_k, 3))

    images = [
        ("original", explanation["x"]),
        ("reconstruction", explanation["reconstruction"]),
        ("residual", np.abs(explanation["residual_image"])),
    ]
    for col, (name, image) in enumerate(images):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image.reshape(image_shape), cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    ax_text = fig.add_subplot(grid[0, 3:])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"index: {int(explanation['index'])}",
                f"label: {float(explanation['label']):.0f}",
                f"prediction: {float(explanation['prediction']):.0f}",
                f"score: {float(explanation['score']):.3f}",
                f"y*score: {float(explanation['signed_margin']):.3f}",
                f"residual: {float(explanation['margin_residual']):.3f}",
                f"violated: {bool(explanation['violated'])}",
            ]
        ),
        va="top",
        family="monospace",
    )

    for col, atom_idx in enumerate(top_atoms):
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(D[:, atom_idx].reshape(image_shape), cmap="gray")
        ax.set_title(
            f"atom {int(atom_idx)}\n"
            f"c={coeffs[col]:.2g}\n"
            f"w*c={contributions[col]:.2g}"
        )
        ax.axis("off")

    fig.tight_layout()
    return fig


def _self_check() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(4, 5))
    D = rng.uniform(0.0, 1.0, size=(4, 3))
    C = rng.normal(size=(3, 5))
    y = np.array([1, -1, 1, -1, 1], dtype=np.float64)
    w = rng.normal(size=3)
    b = 0.1
    diagnostics = margin_diagnostics(C, y, w, b)
    print(select_representative_indices(diagnostics))
    explanation = sample_explanation(X, D, C, y, w, b, 0, top_k=2)
    print(format_sample_explanation(explanation))


if __name__ == "__main__":
    _self_check()
