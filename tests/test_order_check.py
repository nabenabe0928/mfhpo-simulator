from __future__ import annotations

import pytest
import time
import unittest

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import (
    IS_LOCAL,
    OrderCheckConfigs,
    OrderCheckConfigsWithSampleLatency,
    SUBDIR_NAME,
    UNIT_TIME,
    cleanup,
    get_pool,
)


N_EVALS = 20
LATENCY = "latency"
DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    n_workers=2,
    n_actual_evals_in_opt=N_EVALS + 5,
    n_evals=N_EVALS,
)


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
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 190 if latency else 3
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
