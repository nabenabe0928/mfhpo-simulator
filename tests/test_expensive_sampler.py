from __future__ import annotations

import pytest
import time
import unittest
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import ON_UBUNTU, SUBDIR_NAME, cleanup, simplest_dummy_func


class DummyOptimizer:
    def __init__(
        self,
        configs: np.ndarray,
        obj_func: ObjectiveFuncWrapper,
        n_workers: int,
        unittime: float = 1e-3,
    ):
        self._n_workers = n_workers
        self._n_evals = len(configs)
        self._obj_func = obj_func
        self._observations: list[dict[str, float]] = []
        self._unittime = unittime
        self._configs = configs[::-1].tolist()

    def sample(self) -> dict[str, float]:
        waiting_time = (len(self._observations) + 1) * self._unittime
        time.sleep(waiting_time)
        return {"x": self._configs.pop()}

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
                # if len(futures) >= self._n_workers or counts > self._n_evals - self._n_workers:
                if counts >= self._n_workers:
                    self._pop_completed(futures)

                if counts < self._n_evals:
                    config = self.sample()
                    futures[executor.submit(self._obj_func, config)] = config
                    time.sleep(self._unittime * 1e-3)
                    counts += 1


def get_configs(index: int, unittime: float = 1e-3) -> np.ndarray:
    """
    [1] Slow at some points

              |0       |10       |20
              12345678901234567890123456
    Worker 1: sffffssfffff             |
    Worker 2: wsffffffsssfff           |
    Worker 3: wwsffffffwwsssssfff      |
    Worker 4: wwwsfffffwwwwwwwsssssssfff

    [2] Slow from the initialization with correct n_workers
    Usually, it does not work for most optimizers if n_workers is incorrectly set
    because opt libraries typically wait till all the workers are filled up.

              |0       |10       |20
              123456789012345678901234567890
    Worker 1: sfssfwwssssfwwwwssssssf      |
    Worker 2: wsfwsssfwwwsssssfwwwwwsssssssf

    [3] Slow from the initialization with incorrect n_workers ([2] with n_workers=4)
    Assume opt library wait till all the workers are filled up.
    `.` below stands for the waiting time due to the filling up.

              |0       |10       |20
              123456789012345678901234567
    Worker 1: sf..ssssf                 |
    Worker 2: wsf.wwwwsssssf            |
    Worker 3: wwsfwwwwwwwwwssssssf      |
    Worker 4: wwwsfwwwwwwwwwwwwwwsssssssf
    """
    configs = [
        np.array([4.0, 6.0, 6.0, 5.0, 5.0, 3.0, 3.0, 3.0]),
        np.array([0.9] * 8),
        np.array([0.9] * 8),
    ][index]
    ans = [
        np.array([5.0, 8.0, 9.0, 9.0, 12.0, 14.0, 19.0, 26.0]),
        np.array([2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0]),
        np.array([2.0, 3.0, 4.0, 5.0, 9.0, 14.0, 20.0, 27.0]),
    ][index]
    return configs * unittime, ans * unittime


@cleanup
def optimize(index: int, n_workers: int):
    unittime = 1e-1
    configs, ans = get_configs(index=index, unittime=unittime)
    wrapper = ObjectiveFuncWrapper(
        obj_func=simplest_dummy_func,
        n_workers=n_workers,
        n_actual_evals_in_opt=configs.size + 5,
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
    diff = np.abs(np.array(wrapper.get_results()["cumtime"]) - ans)
    assert np.all(diff < unittime * 1.5)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_opt(index: int) -> None:
    if index == 0:
        if ON_UBUNTU:
            optimize(index=index, n_workers=4)
    elif index == 1:
        optimize(index=index, n_workers=2)
    elif index == 2:
        if ON_UBUNTU:
            optimize(index=index, n_workers=4)


if __name__ == "__main__":
    unittest.main()
