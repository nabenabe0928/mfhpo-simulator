from __future__ import annotations

import json
import pytest
import time
import unittest

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import numpy as np

from tests.utils import ON_UBUNTU, OrderCheckConfigs, SUBDIR_NAME, cleanup


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

    xxx means sampling time, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|---|xxx|-------|
              200 300   200 200 200 400
    worker-1: xxx|---|xxx|-------|xxx|---|xxx|-|
              200 200 200 400     200 200 200 100
    """

    def __init__(self, parallel_sampler: bool):
        if parallel_sampler:
            runtimes = np.array([300, 200, 400, 200, 400, 200, 100]) * UNIT_TIME
            self._ans = np.array([400, 500, 900, 1000, 1400, 1500, 1700]) * UNIT_TIME
        else:
            runtimes = np.array([300, 200, 600, 200, 200, 400]) * UNIT_TIME
            self._ans = np.array([500, 600, 1100, 1300, 1500, 1900]) * UNIT_TIME

        loss_vals = [i for i in range(self._ans.size)]
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._n_evals = self._ans.size

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
        ret = dict(index=min(self._count, self._max_count - 1)), None, None
        self._count += 1
        return ret

    def tell(self, *args, **kwargs):
        pass


@cleanup
def optimize_parallel(mode: str, n_workers: int, parallel_sampler: bool = False):
    latency = mode == LATENCY
    kwargs = DEFAULT_KWARGS.copy()
    target = OrderCheckConfigsWithSampleLatency(parallel_sampler) if latency else OrderCheckConfigs(n_workers)
    n_evals = target._n_evals
    kwargs.update(n_workers=n_workers, n_evals=n_evals, allow_parallel_sampling=parallel_sampler)
    wrapper = ObjectiveFuncWrapper(obj_func=target, **kwargs)
    if latency:
        wrapper.simulate(MyOptimizer(UNIT_TIME * 200, max_count=n_evals))
    else:
        wrapper.simulate(MyOptimizer())

    out = json.load(open(wrapper.result_file_path))["cumtime"][:n_evals]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 100 if latency else 1
    assert np.all(diffs < buffer)  # 1 is just a buffer.


@pytest.mark.parametrize("mode", ("normal", LATENCY))
@pytest.mark.parametrize("parallel_sampler", (True, False))
def test_optimize_parallel(mode: str, parallel_sampler: bool):
    if mode == LATENCY:
        optimize_parallel(mode=mode, n_workers=2, parallel_sampler=parallel_sampler)
    elif not parallel_sampler:
        optimize_parallel(mode=mode, n_workers=2)
        optimize_parallel(mode=mode, n_workers=4)
    else:
        pass


if __name__ == "__main__":
    unittest.main()
