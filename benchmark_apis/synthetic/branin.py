from typing import Optional, Tuple

from benchmark_apis.synthetic.abstract_func import MFAbstractFunc

import numpy as np


class MFBranin(MFAbstractFunc):
    """
    Multi-fidelity Branin Function.

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

    def __init__(self, seed: Optional[int] = None, runtime_factor: float = 3600.0):
        super().__init__(seed=seed, runtime_factor=runtime_factor)
        self._noise_std = float(np.sqrt(0.05))
        self._dim = 2

    def _fetch_coefs(self, z: float) -> Tuple[float, float, float, float, float, float]:
        """The coefficients used in Branin. See the reference for more details."""
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin.py#L20-L35
        b0 = 5.1 / (4 * np.pi**2)
        c0 = 5 / np.pi
        t0 = 1 / (8 * np.pi)
        a = 1.0
        b = b0 - 0.01 * (1 - z)
        c = c0 - 0.1 * (1 - z)
        r = 6
        s = 10
        t = t0 + 0.005 * (1 - z)
        return a, b, c, r, s, t

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x *= 15.0
        x[0] -= 5
        return x

    def _objective(self, x: np.ndarray, z: float) -> float:
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin.py#L20-L35
        x = self._transform(x)
        a, b, c, r, s, t = self._fetch_coefs(z)
        x1, x2 = x
        loss = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        noise = self.noise_std * self._rng.normal()
        return float(loss + noise)

    def _runtime(self, x: np.ndarray, z: float) -> float:
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin_mf.py#L24-L26
        runtime = 0.05 + 0.95 * z**1.5
        return float(runtime) * self._runtime_factor
