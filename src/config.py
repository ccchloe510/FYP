"""Central configuration for the FYP experiments."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


DEFAULT_POSITIVE_LABELS: Tuple[int, ...] = (3,)
DEFAULT_NEGATIVE_LABELS: Tuple[int, ...] = (8,)


@dataclass
class DataConfig:
    positive_labels: Tuple[int, ...] = DEFAULT_POSITIVE_LABELS
    negative_labels: Tuple[int, ...] = DEFAULT_NEGATIVE_LABELS
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 400
    normalize: bool = True
    random_state: int = 7


@dataclass(frozen=True)
class TaskConfig:
    """Description of a single binary MNIST task used in the experiments."""

    name: str
    positive_labels: Tuple[int, ...] = DEFAULT_POSITIVE_LABELS
    negative_labels: Optional[Tuple[int, ...]] = DEFAULT_NEGATIVE_LABELS
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 400
    normalize: bool = True
    random_state: int = 7

    def loader_kwargs(self) -> Dict:
        return {
            "positive_labels": self.positive_labels,
            "negative_labels": self.negative_labels,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "normalize": self.normalize,
            "random_state": self.random_state,
        }

    def label_description(self) -> str:
        positive = " + ".join(str(label) for label in self.positive_labels)
        if self.negative_labels is None:
            return f"{positive} vs rest"
        negative = " + ".join(str(label) for label in self.negative_labels)
        return f"{positive} vs {negative}"


@dataclass
class HyperParams:
    dictionary_size: int = 64
    mu: float = 0.05
    rho: float = 1.0
    gamma: float = 0.1
    eta: float = 1.0
    initial_step: float = 1.0
    backtracking_shrink: float = 0.5
    backtracking_min_step: float = 1e-8
    max_iter: int = 100
    tol: float = 1e-5
    random_state: int = 7


def default_data_config() -> DataConfig:
    return DataConfig()


def default_hyperparams() -> HyperParams:
    return HyperParams()


def as_dict(config) -> Dict:
    return asdict(config)


def report_task_suite() -> Tuple[TaskConfig, ...]:
    """Compact binary task suite for the main thesis report."""
    return (
        TaskConfig(name="3 vs 8", positive_labels=(3,), negative_labels=(8,)),
        TaskConfig(name="4 vs 9", positive_labels=(4,), negative_labels=(9,)),
        TaskConfig(name="1 vs 7", positive_labels=(1,), negative_labels=(7,)),
        TaskConfig(name="5 vs 8", positive_labels=(5,), negative_labels=(8,)),
    )


def extended_task_suite() -> Tuple[TaskConfig, ...]:
    """Larger binary suite for stress-testing the joint formulation."""
    return report_task_suite() + (
        TaskConfig(name="0 vs 6", positive_labels=(0,), negative_labels=(6,)),
        TaskConfig(name="2 vs 7", positive_labels=(2,), negative_labels=(7,)),
    )


def one_vs_rest_suite() -> Tuple[TaskConfig, ...]:
    """One-vs-rest supplement for broader MNIST coverage."""
    return tuple(
        TaskConfig(name=f"{digit} vs rest", positive_labels=(digit,), negative_labels=None)
        for digit in range(10)
    )


def task_catalog() -> Dict[str, TaskConfig]:
    """Convenience mapping from task name to task spec."""
    tasks = {}
    for task in extended_task_suite():
        tasks[task.name] = task
    for task in one_vs_rest_suite():
        tasks[task.name] = task
    return tasks


def _self_check() -> None:
    data_cfg = default_data_config()
    hyper = default_hyperparams()
    print("DataConfig:", as_dict(data_cfg))
    print("HyperParams:", as_dict(hyper))
    for task in report_task_suite():
        print("Task:", task.name, task.loader_kwargs())


if __name__ == "__main__":
    _self_check()
