from benchmark_apis.synthetic.abstract_func import MFAbstractFunc

import numpy as np


class TestFunc(MFAbstractFunc):
    def _objective(self, x: np.ndarray, z: float) -> float:
        return float(x[0] ** 2)

    def _runtime(self, x: np.ndarray, z: float):
        return z * 100.0
