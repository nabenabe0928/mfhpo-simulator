import os
import pytest
import shutil
import unittest

from benchmark_simulator.simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper
from benchmark_simulator.utils._performance_over_time import (
    get_mean_and_standard_error,
    get_performance_over_time,
    get_performance_over_time_from_paths,
)

import numpy as np

from tests.utils import dummy_no_fidel_func


class RandomOptimizer(AbstractAskTellOptimizer):
    def __init__(self, seed: int):
        self._rng = np.random.RandomState(seed)

    def ask(self):
        return dict(x=self._rng.random()), None, None

    def tell(self, *args, **kwargs):
        pass


def is_same_interval(x: np.ndarray) -> bool:
    return np.all(np.isclose(x[1:] - x[:-1], x[1] - x[0]))


def is_log_scale(x: np.ndarray) -> bool:
    z = np.log(x)
    return is_same_interval(z)


def _validate(
    n_seeds: int, x: np.ndarray, y: np.ndarray, step: int = 100, log: bool = True, minimize: bool = True
) -> None:
    assert x.shape == (step,)
    assert y.shape == (n_seeds, step)
    if log:
        assert is_log_scale(x)
    else:
        assert is_same_interval(x)

    if minimize:
        assert np.allclose(y, np.minimum.accumulate(y, axis=-1))
    else:
        assert np.allclose(y, np.maximum.accumulate(y, axis=-1))


def run_get_performance_over_time(cumtimes, perf_vals):
    n_seeds = len(cumtimes)
    for kwargs in [dict(), dict(minimize=False), dict(log=False), dict(step=50)]:
        x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=perf_vals, **kwargs)
        _validate(n_seeds, x, y, **kwargs)


def test_cumtime_smaller_than_overhead_error_in_get_performance_over_time():
    rng = np.random.RandomState(0)
    cumtimes = np.sort(rng.random(size=(10, 1000)))
    overheads = cumtimes + 1.0
    with pytest.raises(ValueError, match=r"Each element of optimizer_overheads must be smaller than that of cumtimes."):
        x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=cumtimes.copy(), optimizer_overheads=overheads)


def _errors_in_get_performance_over_time(size1, size2, error, match):
    rng = np.random.RandomState(0)
    cumtimes = np.sort(rng.random(size=size1))
    perf_vals = rng.random(size=size2)
    with pytest.raises(error, match=match):
        run_get_performance_over_time(cumtimes, perf_vals)


def test_2d_array_in_get_performance_over_time():
    n_evals = 1000
    _errors_in_get_performance_over_time(
        size1=n_evals,
        size2=n_evals,
        error=TypeError,
        match=r"cumtimes, perf_vals, and optimizer_overheads must be 2D array*",
    )


def test_diff_seeds_in_get_performance_over_time():
    n_evals = 1000
    _errors_in_get_performance_over_time(
        size1=(10, n_evals),
        size2=(11, n_evals),
        error=ValueError,
        match=r"The number of seeds used in cumtimes, perf_vals, and optimizer_overheads must be identical*",
    )


def test_diff_shapes_in_get_performance_over_time():
    n_evals = 1000
    _errors_in_get_performance_over_time(
        size1=(1, n_evals),
        size2=(1, n_evals + 1),
        error=ValueError,
        match=r"The shapes of cumtimes, perf_vals, and optimizer_overheads for each seed must be identical*",
    )


def test_get_performance_over_time_with_array():
    n_seeds, n_evals = 10, 1000
    rng = np.random.RandomState(0)
    cumtimes = np.sort(rng.random(size=(n_seeds, n_evals)))
    perf_vals = rng.random(size=(n_seeds, n_evals))
    run_get_performance_over_time(cumtimes, perf_vals)


def test_get_performance_over_time_with_list():
    n_seeds, n_evals = 10, 1000
    rng = np.random.RandomState(0)
    cumtimes = [np.sort(rng.random(size=n_evals + i)) for i in range(n_seeds)]
    perf_vals = [rng.random(size=n_evals + i) for i in range(n_seeds)]
    run_get_performance_over_time(cumtimes, perf_vals)


def test_get_performance_over_time_from_paths():
    paths = []
    n_seeds = 10
    for seed in range(n_seeds):
        wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, ask_and_tell=True)
        opt = RandomOptimizer(seed=seed)
        wrapper.simulate(opt)
        paths.append(wrapper.dir_name)
        assert os.path.exists(wrapper.optimizer_overhead_file_path)
        optimizer_overhead = wrapper.get_optimizer_overhead()
        assert len(optimizer_overhead["before_sample"]) >= wrapper._main_wrapper._wrapper_vars.n_evals
        assert len(optimizer_overhead["before_sample"]) == len(optimizer_overhead["after_sample"])

    x1, y1 = get_performance_over_time_from_paths(paths, obj_key="loss")
    x2, y2 = get_performance_over_time_from_paths(paths, obj_key="loss", consider_optimizer_overhead=False)
    assert x1.shape == (100,) and x2.shape == (100,)
    assert y1.shape == (n_seeds, 100) and y2.shape == (n_seeds, 100)
    assert is_log_scale(x1) and is_log_scale(x2)
    assert np.allclose(y1, np.minimum.accumulate(y1, axis=-1)) and np.allclose(y2, np.minimum.accumulate(y2, axis=-1))
    for path in paths:
        shutil.rmtree(path)


def test_get_mean_and_standard_error():
    with pytest.raises(ValueError, match=r"The type of the input must be np.ndarray*"):
        get_mean_and_standard_error(np.arange(6).reshape(2, 3).tolist())
    with pytest.raises(ValueError, match=r"The shape of the input array must be 2D*"):
        get_mean_and_standard_error(np.arange(6).reshape(6))

    x = np.array(
        [
            [1, 2, 3],
            [np.nan, 5, 6],
            [np.nan, np.nan, 8],
        ]
    )
    m, s = get_mean_and_standard_error(x)
    assert np.allclose(m, [1, 7 / 2, 17 / 3])
    assert np.allclose(s, [0.0, 1.5 / np.sqrt(2), 2.0548046676563256 / np.sqrt(3)])


if __name__ == "__main__":
    unittest.main()
