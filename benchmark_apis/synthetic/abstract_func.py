from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np


class MFAbstractFunc(metaclass=ABCMeta):
    """
    Multi-fidelity Function.

    Args:
        seed (Optional[int])
            The random seed for the noise.
        runtime_factor (float):
            The runtime factor to change the maximum runtime.
            If max_fidel is given, the runtime will be the `runtime_factor` seconds.

    Reference:
        Page 18 of the following paper:
            Title: Multi-fidelity Bayesian Optimisation with Continuous Approximations
            Authors: K. Kandasamy et. al
            URL: https://arxiv.org/pdf/1703.06240.pdf
    """

    _DATASET_NAMES = None
    _BENCH_TYPE = "SYNTHETIC"

    def __init__(
        self,
        fidel_dim: int,
        seed: Optional[int] = None,
        runtime_factor: float = 3600.0,
    ):
        if runtime_factor <= 0:
            raise ValueError(f"`runtime_factor` must be positive, but got {runtime_factor}")

        self._rng = np.random.RandomState(seed)
        self._fidel_dim = fidel_dim
        self._runtime_factor = runtime_factor
        self._dim: int
        self._noise_std: float

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fidel_dim(self) -> int:
        return self._fidel_dim

    @property
    def runtime_factor(self) -> float:
        return self._runtime_factor

    @property
    def min_fidels(self) -> Dict[str, Union[float, int]]:
        # the real minimum is 3
        return {f"z{d}": 11 for d in range(self.fidel_dim)}

    @property
    def max_fidels(self) -> Dict[str, Union[float, int]]:
        return {f"z{d}": 100 for d in range(self.fidel_dim)}

    @property
    def fidel_keys(self) -> List[str]:
        return [f"z{d}" for d in range(self.fidel_dim)]

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([CS.UniformFloatHyperparameter(f"x{d}", 0.0, 1.0) for d in range(self._dim)])
        return config_space

    @property
    def noise_std(self) -> float:
        return self._noise_std

    @abstractmethod
    def _objective(self, x: np.ndarray, z: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def _runtime(self, x: np.ndarray, z: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(
        self,
        eval_config: Dict[str, float],
        fidels: Dict[str, int],
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        x = np.array([eval_config[f"x{d}"] for d in range(self._dim)])
        z = np.array([fidels[k] / max_fidel for k, max_fidel in self.max_fidels.items()])
        loss = self._objective(x=x, z=z)
        runtime = self._runtime(x=x, z=z)
        return dict(loss=loss, runtime=runtime)
