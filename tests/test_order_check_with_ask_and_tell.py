from __future__ import annotations

import json
import pytest
import time
import unittest

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import numpy as np

from tests.utils import ON_UBUNTU, SUBDIR_NAME, cleanup


N_EVALS = 20
LATENCY = "latency"
UNIT_TIME = 1e-3 if ON_UBUNTU else 1e-2
DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    ask_and_tell=True,
    n_workers=2,
    n_actual_evals_in_opt=N_EVALS + 5,
    n_evals=N_EVALS,
)


class OrderCheckConfigsWithSampleLatency:
    """
    xxx means sampling time, ooo means waiting time for the sampling for the other worker, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|-----------|xxx|-------|
              200 300   200 600         200 400
    worker-1: ooooxxx|---|ooxxx|---|xxx|---|
              400     200 300   200 200 200
    """

    def __init__(self):
        loss_vals = [i for i in range(6)]
        runtimes = np.array([300, 200, 600, 200, 200, 400]) * UNIT_TIME
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._ans = np.array([500, 600, 1100, 1300, 1500, 1900]) * UNIT_TIME

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


class MyOptimizer(AbstractAskTellOptimizer):
    def __init__(self, sleep: float = 0.0, max_count: int = N_EVALS):
        self._count = 0
        self._max_count = max_count
        self._sleep = sleep

    def ask(self):
        time.sleep(self._sleep)
        ret = dict(index=min(self._count, self._max_count - 1)), None
        self._count += 1
        return ret

    def tell(self, *args, **kwargs):
        pass


@cleanup
def optimize_parallel(mode: str, n_workers: int):
    latency = mode == LATENCY
    kwargs = DEFAULT_KWARGS.copy()
    n_evals = N_EVALS if not latency else 6
    kwargs.update(n_workers=n_workers, n_evals=n_evals)
    target = OrderCheckConfigsWithSampleLatency() if latency else OrderCheckConfigs(n_workers)
    wrapper = ObjectiveFuncWrapper(obj_func=target, **kwargs)
    if latency:
        wrapper.simulate(MyOptimizer(UNIT_TIME * 200, max_count=n_evals))
    else:
        wrapper.simulate(MyOptimizer())

    out = json.load(open(wrapper._main_wrapper._paths.result))["cumtime"][:n_evals]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 100 if latency else 1
    assert np.all(diffs < buffer)  # 1 is just a buffer.


@pytest.mark.parametrize("mode", ("normal", LATENCY))
def test_optimize_parallel(mode):
    if mode == LATENCY:
        optimize_parallel(mode=mode, n_workers=2)
    else:
        optimize_parallel(mode=mode, n_workers=2)
        optimize_parallel(mode=mode, n_workers=4)


if __name__ == "__main__":
    unittest.main()
