from __future__ import annotations

import pytest
import time
import unittest
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

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


@cleanup
def optimize(index: int, n_workers: int):
    unittime = 1e-1 if ON_UBUNTU else 1.0
    configs, ans = get_configs(index=index, unittime=unittime)
    n_evals = configs.size
    wrapper = ObjectiveFuncWrapper(
        obj_func=simplest_dummy_func,
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
    assert np.all(diff < unittime * 1.5)


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 6, 7, 8))
def test_opt(index: int) -> None:
    if index == 1:
        optimize(index=index, n_workers=2)
    elif ON_UBUNTU:
        optimize(index=index, n_workers=4)


if __name__ == "__main__":
    unittest.main()
