from __future__ import annotations

import pytest
import time
import unittest
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import ON_UBUNTU, SUBDIR_NAME, cleanup, get_configs, simplest_dummy_func


class DummyOptimizer:
    def __init__(
        self,
        configs: np.ndarray,
        obj_func: ObjectiveFuncWrapper,
        n_workers: int,
        unittime: float,
    ):
        self._n_workers = n_workers
        self._n_evals = configs.size + self._n_workers
        self._obj_func = obj_func
        self._observations: list[dict[str, float]] = []
        self._unittime = unittime
        self._configs = configs[::-1].tolist()

    def sample(self) -> dict[str, float]:
        waiting_time = (len(self._observations) + 1) * self._unittime
        if len(self._configs):
            time.sleep(waiting_time)
            return {"x": self._configs.pop()}
        else:
            return {"x": 10**5}

    def _pop_completed(self, futures: dict[Future, dict[str, float]]) -> None:
        completed, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in completed:
            config = futures[future]
            try:
                loss = future.result()
            except Exception as e:
                raise RuntimeError(f"An exception occurred: {e}")
            else:
                config["loss"] = loss
                self._observations.append(config.copy())

            futures.pop(future)

    def optimize(self):
        futures: dict[Future, dict[str, float]] = {}
        counts = 0
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            while len(self._observations) < self._n_evals:
                if self._n_workers <= counts <= self._n_evals - self._n_workers or counts == self._n_evals:
                    self._pop_completed(futures)

                if counts < self._n_evals:
                    config = self.sample()
                    futures[executor.submit(self._obj_func, config)] = config
                    time.sleep(self._unittime * 1e-3)
                    counts += 1


def simplest_dummy_func_with_sleep(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    sleeping = 2.0 * (1e-1 if ON_UBUNTU else 1.0)
    time.sleep(sleeping)
    return simplest_dummy_func(eval_config=eval_config, fidels=fidels, seed=seed)


@cleanup
def optimize(index: int, n_workers: int, func: Any):
    unittime = 1e-1 if ON_UBUNTU else 1.0
    configs, ans = get_configs(index=index, unittime=unittime)
    n_evals = configs.size
    wrapper = ObjectiveFuncWrapper(
        obj_func=func,
        n_workers=n_workers,
        n_actual_evals_in_opt=configs.size + n_workers,
        n_evals=configs.size,
        expensive_sampler=True,
        save_dir_name=SUBDIR_NAME,
    )
    opt = DummyOptimizer(
        configs=configs,
        n_workers=wrapper.n_workers,
        obj_func=wrapper,
        unittime=unittime,
    )
    opt.optimize()
    diff = np.abs(np.array(wrapper.get_results()["cumtime"])[:n_evals] - ans)
    print(np.array(wrapper.get_results()["cumtime"])[:n_evals], ans)
    assert np.all(diff < unittime * 1.5)


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 6, 7, 8))
def test_opt(index: int) -> None:
    if index == 1:
        optimize(index=index, n_workers=2, func=simplest_dummy_func)
    elif ON_UBUNTU:
        optimize(index=index, n_workers=4, func=simplest_dummy_func)


@pytest.mark.parametrize("index", (0, 1, 3, 6))
def test_opt_for_long_load(index: int) -> None:
    """
    For example, if a sampling takes 10 seconds,
    a query of a benchmark takes 25 seconds,
    and three workers are free right now at the current actual timestamp of T,
        1. The next sampling for worker 1 starts at T,
        2. The sampling for worker 1 finishes at T + 10,
        3. The query for worker 1 happens at T + 10,
        4. The next sampling for worker 2 starts at T + 10,
        5. The sampling for worker 2 finishes at T + 20,
        6. The sampling for worker 3 starts at T + 20,
        7. The query for worker 1 comes at T + 25 and let's say it says the runtime for this is 3.

    In this case, the result for worker 1 should have come at T + 13 and
    the optimizer should have been able to see the result before the sampling for worker 3,
    but it would not happen.
    Basically, I do not have any rights to stop optimizers from sampling for a worker
    after I release a result of the worker.
    This is not a bug, but this is the software design.
    """
    if index == 1:
        optimize(index=index, n_workers=2, func=simplest_dummy_func)
    elif ON_UBUNTU:
        optimize(index=index, n_workers=4, func=simplest_dummy_func_with_sleep)


if __name__ == "__main__":
    unittest.main()
