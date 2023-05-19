from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

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
        seed: Optional[int] = None,
        runtime_factor: float = 3600.0,
    ):
        if runtime_factor <= 0:
            raise ValueError(f"`runtime_factor` must be positive, but got {runtime_factor}")

        self._rng = np.random.RandomState(seed)
        self._runtime_factor = runtime_factor
        self._dim: int
        self._noise_std: float

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def runtime_factor(self) -> float:
        return self._runtime_factor

    @property
    def min_fidel(self) -> int:
        # the real minimum is 3
        return 11

    @property
    def max_fidel(self) -> int:
        return 100

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([CS.UniformFloatHyperparameter(f"x{d}", 0.0, 1.0) for d in range(self._dim)])
        return config_space

    @property
    def noise_std(self) -> float:
        return self._noise_std

    @abstractmethod
    def _objective(self, x: np.ndarray, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def _runtime(self, x: np.ndarray, z: float) -> float:
        raise NotImplementedError

    def __call__(
        self,
        eval_config: Dict[str, float],
        fidel: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        x = np.array([eval_config[f"x{d}"] for d in range(self._dim)])
        z = fidel / self.max_fidel
        loss = self._objective(x=x, z=z)
        runtime = self._runtime(x=x, z=z)
        return dict(loss=loss, runtime=runtime)
