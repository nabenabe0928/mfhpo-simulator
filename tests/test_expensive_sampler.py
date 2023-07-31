from __future__ import annotations

import time
import unittest
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import simplest_dummy_func


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
        print(f"Sample with {waiting_time=} sec at {time.time()}")
        time.sleep(waiting_time)
        return {"x": self._configs.pop()}

    def _pop_completed(self, futures: dict[Future, dict[str, float]]) -> None:
        completed, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in completed:
            config = futures[future]
            try:
                loss = future.result()
            except Exception as e:
                print(f"An exception occurred: {e}")
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
                    # print(f"Before Pop {len(futures)}")
                    self._pop_completed(futures)
                    # print(f"After Pop {len(futures)}")

                if counts < self._n_evals:
                    # print(f"{len(self._observations)=}")
                    config = self.sample()
                    futures[executor.submit(self._obj_func, config)] = config
                    time.sleep(self._unittime * 1e-3)
                    counts += 1


def get_configs(index: int, unittime: float = 1e-3) -> np.ndarray:
    """
    [2] Slow at some points

              |0       |10       |20
              12345678901234567890123456
    Worker 1: sffffssfffff             |
    Worker 2: wsffffffsssfff           |
    Worker 3: wwsffffffwwsssssfff      |
    Worker 4: wwwsfffffwwwwwwwsssssssfff

    [1] Slow from the initialization

    [3]
    """
    configs = [
        np.array([4.0, 6.0, 6.0, 5.0, 5.0, 3.0, 3.0, 3.0]),
    ][index]
    ans = [
        np.array([5.0, 8.0, 9.0, 9.0, 12.0, 14.0, 19.0, 26.0]),
    ][index]
    return configs * unittime, ans * unittime


def test_opt():
    unittime = 1e-1
    configs, ans = get_configs(index=0, unittime=unittime)
    wrapper = ObjectiveFuncWrapper(
        obj_func=simplest_dummy_func,
        n_workers=4,
        n_actual_evals_in_opt=configs.size + 5,
        n_evals=configs.size,
        expensive_sampler=True,
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


if __name__ == "__main__":
    unittest.main()
