from __future__ import annotations

from typing import ClassVar, Final, Literal

from benchmark_apis.synthetic.abstract_func import MFAbstractFunc

import numpy as np


class MFHartmann(MFAbstractFunc):
    """
    Multi-fidelity Hartmann Function.

    Args:
        dim (Literal[3, 6]):
            The dimension of the search space.
            Either 3 or 6.
        seed (int | None)
            The random seed for the noise.
        bias (float):
            The bias term to change the rank correlation between low- and high-fidelities.
            Larger values lead to less correlation.
            The default value was used in the Dragonfly paper in the reference.
        runtime_factor (float):
            The runtime factor to change the maximum runtime.
            If max_fidel is given, the runtime will be the `runtime_factor` seconds.

    Reference:
        Page 18 of the following paper:
            Title: Multi-fidelity Bayesian Optimisation with Continuous Approximations
            Authors: K. Kandasamy et. al
            URL: https://arxiv.org/pdf/1703.06240.pdf
    """

    _DEFAULT_FIDEL_DIM: Final[ClassVar[int]] = 4

    def __init__(
        self,
        dim: Literal[3, 6],
        fidel_dim: int = 1,
        seed: int | None = None,
        bias: float = 0.1,
        runtime_factor: float = 3600.0,
    ):
        super().__init__(seed=seed, runtime_factor=runtime_factor, fidel_dim=fidel_dim)
        if dim not in [3, 6]:
            self._raise_error_for_wrong_dim()

        self._dim = int(dim)
        noise_var = 0.01 if dim == 3 else 0.05
        self._noise_std = float(np.sqrt(noise_var))
        self._bias = bias

    def _raise_error_for_wrong_dim(self, dim: int) -> None:
        raise ValueError(f"`dim` for Hartmann function must be either 3 or 6, but got {dim}")

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def alphas(self) -> np.ndarray:
        return np.array([1.0, 1.2, 3.0, 3.2])

    @property
    def A(self) -> np.ndarray:
        if self.dim == 3:
            return np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], dtype=np.float64)
        elif self.dim == 6:
            return np.array(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ],
                dtype=np.float64,
            )
        else:
            self._raise_error_for_wrong_dim()

    @property
    def P(self) -> np.ndarray:
        if self.dim == 3:
            return 1e-4 * np.array(
                [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]], dtype=np.float64
            )
        elif self.dim == 6:
            return 1e-4 * np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ],
                dtype=np.float64,
            )
        else:
            self._raise_error_for_wrong_dim()

    def _objective(self, x: np.ndarray, z: np.ndarray) -> float:
        alphas = self.alphas - self._bias * (1 - z)
        loss = -alphas @ np.exp(np.sum(-self.A * (x - self.P) ** 2, axis=-1))
        noise = self.noise_std * self._rng.normal()
        return float(loss + noise)

    def _runtime(self, x: np.ndarray, z: np.ndarray) -> float:
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/hartmann3_2/hartmann3_2_mf.py#L31-L34
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/hartmann6_4/hartmann6_4_mf.py#L27-L30
        z1, z2, z3, z4 = z if self.fidel_dim == self._DEFAULT_FIDEL_DIM else (z[0], z[0], z[0], z[0])
        factor = np.mean([z1, z3 * z4, z2**3]) if self.dim == 3 else np.mean([z1, z3, z2**2, z4**3])
        runtime = 0.1 + 0.9 * factor
        return float(runtime) * self._runtime_factor
