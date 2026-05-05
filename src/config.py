"""Central configuration for the FYP experiments."""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


DEFAULT_LABEL_MAP: Tuple[int, int] = (3, 8)


@dataclass
class DataConfig:
    positive_digit: int = 3
    negative_digit: int = 8
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 400
    normalize: bool = True
    random_state: int = 7


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


def _self_check() -> None:
    data_cfg = default_data_config()
    hyper = default_hyperparams()
    print("DataConfig:", as_dict(data_cfg))
    print("HyperParams:", as_dict(hyper))


if __name__ == "__main__":
    _self_check()
