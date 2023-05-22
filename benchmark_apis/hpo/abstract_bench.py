from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union
import os

import ConfigSpace as CS

import json

import numpy as np


DATA_DIR_NAME = os.path.join(os.environ["HOME"], "tabular_benchmarks")
VALUE_RANGES = json.load(open("benchmark_apis/hpo/discrete_search_spaces.json"))


class AbstractBench(metaclass=ABCMeta):
    _rng: np.random.RandomState
    _value_range: Dict[str, List[Union[int, float, str]]]
    dataset_name: str
    _BENCH_TYPE = "HPO"

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
    def min_fidels(self) -> Dict[str, Union[float, int]]:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        raise NotImplementedError

    @property
    @abstractmethod
    def max_fidels(self) -> Dict[str, Union[float, int]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidel_keys(self) -> List[str]:
        raise NotImplementedError
