from __future__ import annotations

import pytest
import time
import unittest

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import (
    IS_LOCAL,
    OrderCheckConfigs,
    OrderCheckConfigsForSync,
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
def optimize_sync_parallel(mode: str, n_workers: int, sleeping: float = 0.0) -> None:
    latency = mode == LATENCY
    kwargs = DEFAULT_KWARGS.copy()
    if latency:
        # target = OrderCheckConfigsWithSampleLatency(parallel_sampler, timeout)
        pass
    else:
        target = OrderCheckConfigsForSync(n_workers, sleeping=sleeping)

    n_evals = target._n_evals
    kwargs.update(n_workers=n_workers, n_evals=7, n_actual_evals_in_opt=8, batch_size=3)
    wrapper_cls = ObjectiveFuncWrapperWithSampleLatency if latency else ObjectiveFuncWrapper
    wrapper = wrapper_cls(obj_func=target, **kwargs)

    res = [wrapper(eval_config=dict(index=min(index, n_evals - 1))) for index in range(n_evals + 1)]

    out = wrapper.get_results()["cumtime"][:n_evals]
    diffs = out - np.maximum.accumulate(out)
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 100 if latency else 3
    print(target._ans, out)
    assert np.all(diffs < buffer)


@pytest.mark.parametrize("n_workers", (2, 3))
@pytest.mark.parametrize("sleeping", (0.0, UNIT_TIME * 200))
def test_optimize_sync_parallel(n_workers: int, sleeping: float):
    optimize_sync_parallel(mode="normal", n_workers=n_workers, sleeping=sleeping)


@cleanup
def optimize_parallel(mode: str, parallel_sampler: bool, timeout: bool = False, sleeping: float = 0.0) -> None:
    latency = mode == LATENCY
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = 2 if latency or not IS_LOCAL else 4
    if latency:
        target = OrderCheckConfigsWithSampleLatency(parallel_sampler, timeout)
    else:
        target = OrderCheckConfigs(n_workers, sleeping=sleeping)

    n_evals = target._n_evals
    kwargs.update(n_workers=n_workers, n_evals=n_evals, allow_parallel_sampling=parallel_sampler)
    wrapper_cls = ObjectiveFuncWrapperWithSampleLatency if latency else ObjectiveFuncWrapper
    wrapper = wrapper_cls(obj_func=target, **kwargs)

    with get_pool(n_workers=n_workers, join=bool(not timeout)) as pool:
        res = []
        for index in range(n_evals + n_workers):
            if timeout:
                time.sleep(UNIT_TIME * 200)
                if index > 1:
                    res[-1].get()

            r = pool.apply_async(wrapper, kwds=dict(eval_config=dict(index=min(index, n_evals - 1))))
            res.append(r)
        for r in res:
            r.get()

    out = wrapper.get_results()["cumtime"][:n_evals]
    diffs = out - np.maximum.accumulate(out)
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 100 if latency else 3
    assert np.all(diffs < buffer)


@pytest.mark.parametrize("mode", ("normal", LATENCY))
@pytest.mark.parametrize("sleeping", (0.0, UNIT_TIME * 200))
@pytest.mark.parametrize("parallel_sampler", (True, False))
def test_optimize_parallel(mode: str, sleeping: float, parallel_sampler: bool):
    if mode == "normal" and parallel_sampler:
        pytest.skip("No need to consider parallel sampler for cheap benchmark.")

    if mode == LATENCY and sleeping > 0.0:
        pytest.skip("If benchmark has overhead, we cannot handle expensive sampler.")

    optimize_parallel(mode=mode, parallel_sampler=parallel_sampler, sleeping=sleeping)


@cleanup
def test_opt_init_timeout():
    with pytest.raises(TimeoutError, match=r"The initialization of the optimizer must be cheaper*"):
        optimize_parallel(mode=LATENCY, parallel_sampler=False, timeout=True)


if __name__ == "__main__":
    unittest.main()
