from __future__ import annotations

import pytest
import time
import unittest

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import IS_LOCAL, ON_UBUNTU, OrderCheckConfigs, SUBDIR_NAME, cleanup, get_pool


N_EVALS = 20
LATENCY = "latency"
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

    xxx means sampling time, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: -----|xxx|---|xxx|-------|
              300   200 200 200 400
    worker-1: ---|xxx|-------|xxx|---|xxx|-|
              200 200 400     200 200 200 100
    """

    def __init__(self, parallel_sampler: bool):
        if parallel_sampler:
            runtimes = np.array([300, 200, 400, 200, 400, 200, 100]) * UNIT_TIME
            self._ans = np.array([200, 300, 700, 800, 1200, 1300, 1500]) * UNIT_TIME
        else:
            runtimes = np.array([500, 600, 600, 200, 200, 400, 200, 200, 200]) * UNIT_TIME
            self._ans = np.array([500, 600, 1100, 1300, 1500, 1900, 1900, 2300, 2500]) * UNIT_TIME

        loss_vals = [i for i in range(self._ans.size)]
        self._results = [dict(loss=loss, runtime=runtime) for loss, runtime in zip(loss_vals, runtimes)]
        self._n_evals = self._ans.size

    def __call__(self, eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
        results = self._results[eval_config["index"]]
        return results


class ObjectiveFuncWrapperWithSampleLatency(ObjectiveFuncWrapper):
    def __call__(self, eval_config, **kwargs):
        time.sleep(UNIT_TIME * 200)
        super().__call__(eval_config, **kwargs)


@cleanup
def optimize_parallel(mode: str, parallel_sampler: bool):
    latency = mode == LATENCY
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = 2 if latency or not IS_LOCAL else 4
    target = OrderCheckConfigsWithSampleLatency(parallel_sampler) if latency else OrderCheckConfigs(n_workers)
    n_evals = target._n_evals
    kwargs.update(n_workers=n_workers, n_evals=n_evals, allow_parallel_sampling=parallel_sampler)
    wrapper_cls = ObjectiveFuncWrapperWithSampleLatency if latency else ObjectiveFuncWrapper
    wrapper = wrapper_cls(obj_func=target, **kwargs)

    with get_pool(n_workers=n_workers) as pool:
        res = []
        for index in range(n_evals + n_workers):
            r = pool.apply_async(wrapper, kwds=dict(eval_config=dict(index=min(index, n_evals - 1))))
            res.append(r)
        for r in res:
            r.get()

    out = wrapper.get_results()["cumtime"][:n_evals]
    diffs = out - np.maximum.accumulate(out)
    assert np.allclose(diffs, 0.0)
    diffs = out - target._ans
    buffer = UNIT_TIME * 100 if latency else 3
    assert np.all(diffs < buffer)


@pytest.mark.parametrize("mode", ("normal", LATENCY))
@pytest.mark.parametrize("parallel_sampler", (True, False))
def test_optimize_parallel(mode: str, parallel_sampler: bool):
    if mode == "normal" and parallel_sampler:
        # No test
        return

    optimize_parallel(mode=mode, parallel_sampler=parallel_sampler)


if __name__ == "__main__":
    unittest.main()
