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
