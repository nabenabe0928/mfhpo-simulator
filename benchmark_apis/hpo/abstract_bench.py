from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Final
import os

import ConfigSpace as CS

import json

import numpy as np


DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "tabular_benchmarks")
SEARCH_SPACE_PATH: Final[str] = "benchmark_apis/hpo/discrete_search_spaces.json"
VALUE_RANGES: Final[dict[str, list[int | float | str | bool]]] = json.load(open(SEARCH_SPACE_PATH))


class AbstractBench(metaclass=ABCMeta):
    _BENCH_TYPE: Final[ClassVar[str]] = "HPO"
    _target_metric: ClassVar[str]
    _value_range: ClassVar[dict[str, list[int | float | str | bool]]]
    _rng: np.random.RandomState
    dataset_name: str

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @abstractmethod
    def get_benchdata(self) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_fidels(self) -> dict[str, int | float]:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        raise NotImplementedError

    @property
    @abstractmethod
    def max_fidels(self) -> dict[str, int | float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidel_keys(self) -> list[str]:
        raise NotImplementedError
