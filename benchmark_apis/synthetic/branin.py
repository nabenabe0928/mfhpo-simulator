from __future__ import annotations

from dataclasses import dataclass

from typing import ClassVar, Final

from benchmark_apis.synthetic.abstract_func import MFAbstractFunc

import numpy as np


@dataclass(frozen=True)
class _BraninCoefficients:
    a: float
    b: float
    c: float
    r: float
    s: float
    t: float


class MFBranin(MFAbstractFunc):
    """
    Multi-fidelity Branin Function.

    Args:
        delta_b, delta_c, delta_t (float):
            The control parameters of the rank correlation between low- and high-fidelities.
            Larger values lead to less correlation.
            The default value was used in the Dragonfly paper in the reference.
        seed (int | None)
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

    _DEFAULT_FIDEL_DIM: Final[ClassVar[int]] = 3

    def __init__(
        self,
        fidel_dim: int = 1,
        delta_b: float = 1e-2,
        delta_c: float = 0.1,
        delta_t: float = 5e-3,
        seed: int | None = None,
        runtime_factor: float = 3600.0,
    ):
        super().__init__(fidel_dim=fidel_dim, seed=seed, runtime_factor=runtime_factor)
        self._noise_std = float(np.sqrt(0.05))
        self._dim = 2
        self._delta_b, self._delta_c, self._delta_t = delta_b, delta_c, delta_t

    def _fetch_coefs(self, z: np.ndarray) -> _BraninCoefficients:
        """The coefficients used in Branin. See the reference for more details."""
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin.py#L20-L35
        z1, z2, z3 = z if self.fidel_dim == self._DEFAULT_FIDEL_DIM else (z[0], z[0], z[0])
        b0 = 5.1 / (4 * np.pi**2)
        c0 = 5 / np.pi
        t0 = 1 / (8 * np.pi)
        return _BraninCoefficients(
            a=1.0,
            b=b0 - self._delta_b * (1 - z1),
            c=c0 - self._delta_c * (1 - z2),
            r=6,
            s=10,
            t=t0 + self._delta_t * (1 - z3),
        )

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x *= 15.0
        x[0] -= 5
        return x

    def _objective(self, x: np.ndarray, z: np.ndarray) -> float:
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin.py#L20-L35
        x = self._transform(x)
        coefs = self._fetch_coefs(z)
        x1, x2 = x
        loss = (
            coefs.a * (x2 - coefs.b * x1**2 + coefs.c * x1 - coefs.r) ** 2
            + coefs.s * (1 - coefs.t) * np.cos(x1)
            + coefs.s
        )
        noise = self.noise_std * self._rng.normal()
        return float(loss + noise)

    def _runtime(self, x: np.ndarray, z: np.ndarray) -> float:
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin_mf.py#L24-L26
        z1 = z[0]
        runtime = 0.05 + 0.95 * z1**1.5
        return float(runtime) * self._runtime_factor
