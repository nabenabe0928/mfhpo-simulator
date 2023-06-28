from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import time
import unittest

from benchmark_simulator import ObjectiveFuncWrapper
from benchmark_simulator._constants import DIR_NAME

import numpy as np


SUBDIR_NAME = "dummy"
IS_LOCAL = eval(os.environ.get("MFHPO_SIMULATOR_TEST", "False"))
ON_UBUNTU = sys.platform == "linux"
PATH = os.path.join(DIR_NAME, SUBDIR_NAME)
N_EVALS = 20
UNIT_TIME = 1e-3 if ON_UBUNTU else 1e-2
DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    n_workers=2,
    n_actual_evals_in_opt=N_EVALS + 5,
    n_evals=N_EVALS,
)


class OrderCheckConfigsWithSampleLatency:
    """
    xxx means sampling time, ooo means waiting time for the sampling for the other worker, --- means waiting time.
    Note that the first sample cannot be considered.

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: ---------|xxx|-----------|xxx|-------|ooooxxx|---|
              500       200 600         200 400     400     200
    worker-1: -----------|ooxxx|---|xxx|---|xxx|---|xxx|---|
              600         300   200 200 200 200 200 200 200
    """

    def __init__(self):
        loss_vals = [i for i in range(9)]
        runtimes = np.array([500, 600, 600, 200, 200, 400, 200, 200, 200]) * UNIT_TIME
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._ans = np.array([500, 600, 1100, 1300, 1500, 1900, 1900, 2300, 2500]) * UNIT_TIME

    def __call__(self, eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
        results = self._results[eval_config["index"]]
        return results


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
        loss_vals = [i for i in range(N_EVALS)]
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

    def __call__(self, eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
        results = self._results[eval_config["index"]]
        return results


def remove_tree():
    try:
        shutil.rmtree(PATH)
    except FileNotFoundError:
        pass


def optimize_parallel(n_workers: int):
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = n_workers
    target = OrderCheckConfigs(n_workers)
    manager = ObjectiveFuncWrapper(obj_func=target, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for index in range(N_EVALS + 4):
        r = pool.apply_async(manager, kwds=dict(eval_config=dict(index=min(index, N_EVALS - 1))))
        res.append(r)
    else:
        for r in res:
            r.get()

    pool.close()
    pool.join()

    path = manager.dir_name
    out = json.load(open(os.path.join(path, "results.json")))["cumtime"][:N_EVALS]
    shutil.rmtree(path)
    diffs = out - np.maximum.accumulate(out)
    assert np.allclose(diffs, 0.0)
    diffs = out - target._ans
    assert np.all(diffs < 3)  # 3 is just a buffer.


def test_optimize_parallel():
    if IS_LOCAL:
        optimize_parallel(n_workers=4)

    optimize_parallel(n_workers=2)


class ObjectiveFuncWrapperWithSampleLatency(ObjectiveFuncWrapper):
    def __call__(self, eval_config, **kwargs):
        time.sleep(UNIT_TIME * 200)
        super().__call__(eval_config, **kwargs)


def test_optimize_with_latency():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers, n_evals = 2, 9
    kwargs["n_evals"] = n_evals
    kwargs["n_workers"] = n_workers
    target = OrderCheckConfigsWithSampleLatency()
    manager = ObjectiveFuncWrapperWithSampleLatency(obj_func=target, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for index in range(n_evals + n_workers):
        eval_config = dict(index=min(index, n_evals - 1))
        r = pool.apply_async(manager, kwds=dict(eval_config=eval_config))
        res.append(r)
    else:
        for r in res:
            r.get()

    pool.close()
    pool.join()

    path = manager.dir_name
    out = json.load(open(os.path.join(path, "results.json")))["cumtime"][:9]
    shutil.rmtree(path)
    diffs = out - np.maximum.accumulate(out)
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    assert np.all(diffs < UNIT_TIME * 100)  # Right hand side is not zero, because need some buffer.


if __name__ == "__main__":
    unittest.main()
