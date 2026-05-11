"""Dataset loading and preprocessing utilities."""

import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - informative runtime failure
    raise ImportError(
        "scikit-learn is required for src.data. Install requirements.txt first."
    ) from exc


ArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
LabelSpec = Union[int, Sequence[int]]
DEFAULT_LOCAL_MNIST = Path.home() / ".keras" / "datasets" / "mnist.npz"


def _validate_split_sizes(total_requested: int, available: int) -> None:
    if total_requested > available:
        raise ValueError(
            f"Requested {total_requested} samples but only {available} are available."
        )


def _as_label_tuple(labels: LabelSpec) -> Tuple[int, ...]:
    if isinstance(labels, (int, np.integer)):
        return (int(labels),)
    normalized = tuple(int(label) for label in labels)
    if not normalized:
        raise ValueError("label specification cannot be empty")
    return normalized


def _resolve_negative_labels(
    y_raw: np.ndarray,
    positive_labels: Tuple[int, ...],
    negative_labels: Union[None, LabelSpec],
) -> Tuple[int, ...]:
    if negative_labels is None:
        all_labels = {int(label) for label in np.unique(y_raw)}
        resolved = tuple(sorted(all_labels.difference(set(positive_labels))))
        if not resolved:
            raise ValueError("positive_labels already cover all classes; no negative labels remain")
        return resolved
    return _as_label_tuple(negative_labels)


def _validate_binary_label_sets(
    positive_labels: Tuple[int, ...],
    negative_labels: Tuple[int, ...],
) -> None:
    overlap = set(positive_labels).intersection(negative_labels)
    if overlap:
        raise ValueError(f"positive_labels and negative_labels must be disjoint, got overlap={sorted(overlap)}")


def _load_local_mnist_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        X_train = data["x_train"].reshape(data["x_train"].shape[0], -1)
        y_train = data["y_train"]
        X_test = data["x_test"].reshape(data["x_test"].shape[0], -1)
        y_test = data["y_test"]
    X = np.concatenate([X_train, X_test], axis=0).astype(np.float64)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    return X, y


def _load_mnist_arrays() -> Tuple[np.ndarray, np.ndarray]:
    local_path = os.environ.get("MNIST_NPZ_PATH")
    if local_path:
        candidate = Path(local_path).expanduser()
        if candidate.exists():
            return _load_local_mnist_npz(candidate)

    if DEFAULT_LOCAL_MNIST.exists():
        return _load_local_mnist_npz(DEFAULT_LOCAL_MNIST)

    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    except Exception as exc:  # pragma: no cover - network and local environment dependent
        raise RuntimeError(
            "Unable to load MNIST. No local mnist.npz was found and OpenML download failed. "
            "Place a local file at ~/.keras/datasets/mnist.npz or set MNIST_NPZ_PATH."
        ) from exc

    return mnist.data.astype(np.float64), mnist.target.astype(np.int64)


def load_mnist_binary(
    positive_labels: LabelSpec = (3,),
    negative_labels: Union[None, LabelSpec] = (8,),
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 400,
    normalize: bool = True,
    random_state: int = 7,
) -> ArrayTuple:
    """Load MNIST and return column-major matrices for a binary task.

    Examples:
    - 3 vs 8: positive_labels=3, negative_labels=8
    - 1 vs 9: positive_labels=[1], negative_labels=[9]
    - 1 vs rest: positive_labels=1, negative_labels=None
    - merged classes: positive_labels=[3, 5], negative_labels=[8, 9]
    """
    X, y_raw = _load_mnist_arrays()

    positive_labels = _as_label_tuple(positive_labels)
    negative_labels = _resolve_negative_labels(y_raw, positive_labels, negative_labels)
    _validate_binary_label_sets(positive_labels, negative_labels)

    positive_mask = np.isin(y_raw, positive_labels)
    negative_mask = np.isin(y_raw, negative_labels)
    mask = positive_mask | negative_mask
    X = X[mask]
    y_raw = y_raw[mask]
    y = np.where(np.isin(y_raw, positive_labels), 1.0, -1.0)

    total_requested = train_size + val_size + test_size
    _validate_split_sizes(total_requested, X.shape[0])

    X_selected, _, y_selected, _ = train_test_split(
        X,
        y,
        train_size=total_requested,
        stratify=y,
        random_state=random_state,
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected,
        y_selected,
        test_size=test_size,
        stratify=y_selected,
        random_state=random_state,
    )
    train_fraction = train_size / float(train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        train_size=train_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    if normalize:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0

    return (
        X_train.T,
        y_train.astype(np.float64),
        X_val.T,
        y_val.astype(np.float64),
        X_test.T,
        y_test.astype(np.float64),
    )



def _self_check() -> None:
    print(f"DEFAULT_LOCAL_MNIST exists: {DEFAULT_LOCAL_MNIST.exists()}")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_binary(
            positive_labels=3,
            negative_labels=8,
            train_size=20,
            val_size=10,
            test_size=10,
        )
        print(
            "Loaded shapes:",
            X_train.shape,
            y_train.shape,
            X_val.shape,
            y_val.shape,
            X_test.shape,
            y_test.shape,
        )
        print("Value range:", float(X_train.min()), float(X_train.max()))
        X_train, y_train, _, _, _, _ = load_mnist_binary(
            positive_labels=1,
            negative_labels=None,
            train_size=20,
            val_size=10,
            test_size=10,
        )
        print("1-vs-rest sample shapes:", X_train.shape, y_train.shape)
    except Exception as exc:
        print(f"Data self-check skipped: {exc}")


if __name__ == "__main__":
    _self_check()
