from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
from contextlib import contextmanager
from typing import Any

from benchmark_simulator import ObjectiveFuncWrapper
from benchmark_simulator._constants import DIR_NAME

import numpy as np


SUBDIR_NAME = "dummy"
SIMPLE_CONFIG = {"x": 0}
IS_LOCAL = eval(os.environ.get("MFHPO_SIMULATOR_TEST", "False"))
ON_UBUNTU = sys.platform == "linux"
UNIT_TIME = 1e-3 if ON_UBUNTU else 5e-2
DIR_PATH = os.path.join(DIR_NAME, SUBDIR_NAME)


class OrderCheckConfigs:
    """
    [1] 2 worker case
    worker-0: -------------------|-|---|---|---|---|---|---|---|---|-----|
              1000              100 200 200 200 200 200 200 200 200 300
    worker-1: -----|-----|-----|-----|-----------|-----|---|-----------|-------|
              300   300   300   300   600         300   200 600         400

    [2] 4 worker case
    worker-0: -------------------|-----|-----|
              1000                300   300
    worker-1: -------|-------|-------|-------|
              400     400     400     400
    worker-2: -----|-----|-----|-----|-----|
              300   300   300   300   300
    worker-3: ---|---|---|---|---|---|---|-|
              200 200 200 200 200 200 200 100
    """

    def __init__(self, n_workers: int):
        loss_vals = [i for i in range(20)]
        runtimes = {
            2: [1000, 300, 300, 300, 300, 100, 200, 600, 200, 200, 200, 300, 200, 200, 200, 600, 200, 200, 300, 400],
            4: [1000, 400, 300, 200, 200, 300, 400, 200, 200, 300, 200, 400, 300, 200, 300, 200, 300, 400, 300, 100],
        }[n_workers]
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._ans = {
            2: np.array(
                [
                    300,
                    600,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                    1500,
                    1700,
                    1800,
                    1900,
                    2100,
                    2100,
                    2300,
                    2300,
                    2500,
                    2700,
                    2900,
                    3000,
                    3300,
                ]
            ),
            4: np.array(
                [
                    200,
                    300,
                    400,
                    400,
                    600,
                    600,
                    800,
                    800,
                    900,
                    1000,
                    1000,
                    1200,
                    1200,
                    1200,
                    1300,
                    1400,
                    1500,
                    1500,
                    1600,
                    1600,
                ]
            ),
        }[n_workers]
        self._n_evals = self._ans.size

    def __call__(self, eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
        results = self._results[eval_config["index"]]
        return results


class OrderCheckConfigsWithSampleLatency:
    """
    xxx means sampling time, ooo means waiting time for the sampling for the other worker, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!
    NOTE: I supported first sample consideration for the non ask-and-tell version as well.

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|-----------|xxx|-------|
              200 300   200 600         200 400
    worker-1: ooooxxx|---|ooxxx|---|xxx|---|
              400     200 300   200 200 200

    [2] 2 worker case for Timeout (sampling time is 200 ms)
    worker-0: xxx|-|
              200 100
    worker-1: ooooxxx|
              400

    xxx means sampling time, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|---|xxx|-------|
              200 300   200 200 200 400
    worker-1: xxx|---|xxx|-------|xxx|---|xxx|-|
              200 200 200 400     200 200 200 100
    """

    def __init__(self, parallel_sampler: bool, timeout: bool = False):
        if parallel_sampler and not timeout:
            runtimes = np.array([300, 200, 400, 200, 400, 200, 100]) * UNIT_TIME
            self._ans = np.array([400, 500, 900, 1000, 1400, 1500, 1700]) * UNIT_TIME
        elif not parallel_sampler and timeout:
            runtimes = np.array([100] * 4) * UNIT_TIME
            self._ans = np.array([np.nan] * 4) * UNIT_TIME
        else:
            runtimes = np.array([300, 200, 600, 200, 200, 400]) * UNIT_TIME
            self._ans = np.array([500, 600, 1100, 1300, 1500, 1900]) * UNIT_TIME

        loss_vals = [i for i in range(self._ans.size)]
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._n_evals = self._ans.size

    def __call__(self, eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
        results = self._results[eval_config["index"]]
        return results


def get_configs(index: int, unittime: float) -> np.ndarray:
    """
    [0] Slow at some points

              |0       |10       |20
              12345678901234567890123456
    Worker 1: sffffssfffff             |
    Worker 2: wsffffffsssfff           |
    Worker 3: wwsffffffwwsssssfff      |
    Worker 4: wwwsfffffwwwwwwwsssssssfff

    [1] Slow from the initialization with correct n_workers
    Usually, it does not work for most optimizers if n_workers is incorrectly set
    because opt libraries typically wait till all the workers are filled up.

              |0       |10       |20
              123456789012345678901234567890
    Worker 1: sfssfwwssssfwwwwssssssf      |
    Worker 2: wsfwsssfwwwsssssfwwwwwsssssssf

    [2] Slow from the initialization with incorrect n_workers ([2] with n_workers=4)
    Assume opt library wait till all the workers are filled up.
    `.` below stands for the waiting time due to the filling up.

              |0       |10       |20
              123456789012345678901234567
    Worker 1: sf..ssssf                 |
    Worker 2: wsf.wwwwsssssf            |
    Worker 3: wwsfwwwwwwwwwssssssf      |
    Worker 4: wwwsfwwwwwwwwwwwwwwsssssssf

    [3] No overlap

              |0       |10       |20
              1234567890123456789012345678
    Worker 1: sfffffssfffffffffffff      |
    Worker 2: wsfffffffffffffssssff      |
    Worker 3: wwsffffffffsssfffffff      |
    Worker 4: wwwsffffffffffffffffsssssfff

    The random cases were generated by:
    ```python
    size = np.random.randint(15) + 4
    print((np.random.randint(6, size=size) + 1).tolist())
    ```
    Note that I manually adapt costs of each call if their ends overlap with a start of sampling.
    It is necessary to make the test results more or less deterministic.

    The random cases were visualized with the following code:
    ```python
    import numpy as np

    def func(seq: np.ndarray):
        workers = [0, 0, 0, 0]
        strings = ["", "", "", ""]
        past = []
        before = 0
        for i, v in enumerate(seq):
            min_cumtime = min(workers)
            idx = workers.index(min_cumtime)
            before = max(workers[idx], before)
            s = sum(p <= before for p in past) + 1 if i >= 4 else 1
            strings[idx] += "w" * (before - workers[idx])
            strings[idx] += "s" * s
            strings[idx] += "f" * v
            workers[idx] = before + s + v
            before = workers[idx] - v
            past.append(workers[idx])

        for s in strings:
            print(s)

        print(np.sort(past).tolist())
    ```

    [4] Random case 1

              |0       |10       |20       |30       |40       |50       |60       |70
              123456789012345678901234567890123456789012345678901234567890123456789012345
    Worker 1: sfwwssffffffwwwwwwwwwwwwwwwwwwsssssssssfffff                              |
    Worker 2: wsfffwssssfwwwwwwwwwwwssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfff
    Worker 3: wwsffffwwwwwwwwsssssssffwwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssff            |
    Worker 4: wwwsffwwwwsssssfwwwwwwwwwwwwwwwwwwwwwwwssssssssssff                       |

    [5] Random case 2

              |0       |10       |20       |30       |40
              1234567890123456789012345678901234567890123
    Worker 1: sfwwssffwwssssssffffff                    |
    Worker 2: wsfffffsssfffwwwwwwwwwwwwwwwwwwsssssssssfff
    Worker 3: wwsfffffwwwwwwwwsssssssff                 |
    Worker 4: wwwsfffffwwwwwwwwwwwwwwssssssssfff        |

    [6] Random case 3

              |0       |10       |20       |30       |40       |50       |60       |70
              12345678901234567890123456789012345678901234567890123456789012345678901234
    Worker 1: sfffssfffffwwwwwwwwwwssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfff
    Worker 2: wsffffffwwsssssfffffwwwwwwwwwwwwwwwwwwssssssssssffff                     |
    Worker 3: wwsffffsssffffffwwwwwwwwwwwwwsssssssssffff                               |
    Worker 4: wwwsffffffwwwwwssssssffffwwwwwwwwwwwwwwwwwwwwwwwsssssssssssfffff         |

    [7] Random case 4

              |0       |10       |20       |30       |40       |50       |60       |70       |80       |90       |100
              12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901
    Worker 1: sffffssffffffwwwwwwwwwssssssssffffwwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfffff                       |
    Worker 2: wsffffffwwsssssffffffwwwwwwwwwwwwwwwwwwssssssssssffffwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssssff
    Worker 3: wwsfffwsssffffwwwwwwwwwwwwwwwwsssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssssff             |
    Worker 4: wwwsfffffwwwwwwsssssssfffwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssff                                      |

    [8] Random case 5

              |0       |10       |20
              123456789012345678901234567
    Worker 1: sffffwssssf               |
    Worker 2: wsffffwwwwsssssffffff     |
    Worker 3: wwsfssfffff               |
    Worker 4: wwwsffffwwwwwwwsssssssfffff
    """
    configs = [
        np.array([4, 6, 6, 5, 5, 3, 3, 3], dtype=np.float64),
        np.array([0.9] * 8, dtype=np.float64),
        np.array([0.9] * 8, dtype=np.float64),
        np.array([5, 13, 8, 16, 13, 7, 2, 3], dtype=np.float64),
        np.array([1, 3, 4, 2, 6, 1, 1, 2, 5, 5, 2, 2, 3], dtype=np.float64),
        np.array([1, 5, 5, 5, 2, 3, 6, 2, 3, 3], dtype=np.float64),
        np.array([3, 6, 4, 6, 5, 6, 5, 4, 5, 4, 4, 5, 3], dtype=np.float64),
        np.array([4, 6, 3, 5, 6, 4, 6, 3, 4, 5, 4, 2, 5, 2, 2], dtype=np.float64),
        np.array([4, 4, 1, 4, 5, 1, 6, 5], dtype=np.float64),
    ][index]
    ans = [
        np.array([5, 8, 9, 9, 12, 14, 19, 26], dtype=np.float64),
        np.array([2, 3, 5, 8, 12, 17, 23, 30], dtype=np.float64),
        np.array([2, 3, 4, 5, 9, 14, 20, 27], dtype=np.float64),
        np.array([6, 11, 15, 20, 21, 21, 21, 28], dtype=np.float64),
        np.array([2, 5, 6, 7, 11, 12, 16, 24, 35, 44, 51, 62, 75], dtype=np.float64),
        np.array([2, 7, 8, 8, 9, 13, 22, 25, 34, 43], dtype=np.float64),
        np.array([4, 7, 8, 10, 11, 16, 20, 25, 34, 42, 52, 64, 74], dtype=np.float64),
        np.array([5, 6, 8, 9, 13, 14, 21, 25, 34, 44, 53, 62, 77, 87, 101], dtype=np.float64),
        np.array([4, 5, 6, 8, 11, 11, 21, 27], dtype=np.float64),
    ][index]
    return configs * unittime, ans * unittime


def dummy_func(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def simplest_dummy_func(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    return dict(loss=eval_config["x"], runtime=eval_config["x"])


def dummy_func_with_constant_runtime(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    return dict(loss=eval_config["x"], runtime=1)


def dummy_no_fidel_func(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    return dict(loss=eval_config["x"], runtime=10)


def dummy_func_with_many_fidelities(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
    **data_to_scatter: Any,
) -> dict[str, float]:
    runtime = fidels["z1"] + fidels["z2"] + fidels["z3"]
    return dict(loss=eval_config["x"], runtime=runtime)


def cleanup(test_func) -> None:
    def _inner_func(**kwargs):
        remove_tree()
        test_func(**kwargs)
        remove_tree()

    return _inner_func


def get_results(*, pool, func, n_configs: int, epoch_func: callable, x_func: callable, conditional=None):
    res = {}
    for i in range(n_configs):
        eval_config = {"x": x_func(i)} if conditional is None else conditional(i)
        kwargs = dict(
            eval_config=eval_config,
            fidels={"epoch": epoch_func(i)},
        )
        r = pool.apply_async(func, kwds=kwargs)
        res[i] = r

    return res


def remove_tree():
    try:
        shutil.rmtree(DIR_PATH)
    except FileNotFoundError:
        pass


def get_worker_wrapper(**kwargs):
    return ObjectiveFuncWrapper(**kwargs, launch_multiple_wrappers_from_user_side=True)


def get_n_workers():
    n_workers = 4 if IS_LOCAL else 2  # github actions has only 2 cores
    return n_workers


@contextmanager
def get_pool(n_workers: int, join: bool = True) -> multiprocessing.Pool:
    pool = multiprocessing.Pool(processes=n_workers)
    yield pool
    pool.close()
    if join:  # we do not join if one of the workers is terminated because the info cannot be joined.
        pool.join()
