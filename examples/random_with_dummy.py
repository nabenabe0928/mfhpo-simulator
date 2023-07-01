from __future__ import annotations

import multiprocessing
from typing import Any

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np


class RandomOptimizer:
    def __init__(self, func: callable, n_workers: int, n_evals: int, value_range: tuple[float, float], seed: int = 42):
        self._func = func
        self._n_workers = n_workers
        self._n_evals = n_evals
        self._rng = np.random.RandomState(seed)
        self._lower, self._upper = value_range

    def optimize(self) -> list[float]:
        pool = multiprocessing.Pool(processes=self._n_workers)

        _results = []
        for i in range(self._n_evals):
            x = self._rng.random() * (self._upper - self._lower) + self._lower
            _results.append(pool.apply_async(self._func, args=[x]))

        pool.close()
        pool.join()
        return [r.get() for r in _results]


def dummy_func(x: float) -> float:
    return x**2


def dummy_func_wrapper(eval_config: dict[str, Any], **kwargs) -> dict[str, float]:
    # 0. Adapt the function signature to our wrapper interface
    loss = dummy_func(x=eval_config["x"])
    actual_runtime = loss * 1e3
    # Default: obj_keys = ["loss"], runtime_key = "runtime"
    # You can add more keys to obj_keys then our wrapper collects these values as well.
    return dict(loss=loss, runtime=actual_runtime)


class MyObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # 0. Adapt the callable of the objective function to RandomOptimizer interface
    def __call__(self, x: float) -> float:
        results = super().__call__(eval_config={"x": x})
        return results[self.obj_keys[0]]


if __name__ == "__main__":
    # 1. Define a wrapper instance (Default is n_workers=4, but you can change it from the argument)
    wrapper = MyObjectiveFuncWrapper(obj_func=dummy_func_wrapper)

    RandomOptimizer(
        # 2. Feed the wrapped objective function to the optimizer directly
        func=wrapper,
        n_workers=wrapper.n_workers,
        n_evals=wrapper.n_actual_evals_in_opt,
        value_range=(-5.0, 5.0),
    ).optimize()  # 3. just start the optimization!
