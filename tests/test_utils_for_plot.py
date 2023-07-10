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


def run_get_performance_over_time(cumtimes, perf_vals):
    n_seeds = len(cumtimes)
    x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=perf_vals)
    assert x.shape == (100,)
    assert y.shape == (n_seeds, 100)
    assert is_log_scale(x)
    assert np.allclose(y, np.minimum.accumulate(y, axis=-1))

    x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=perf_vals, minimize=False)
    assert x.shape == (100,)
    assert y.shape == (n_seeds, 100)
    assert is_log_scale(x)
    assert np.allclose(y, np.maximum.accumulate(y, axis=-1))

    x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=perf_vals, log=False)
    assert x.shape == (100,)
    assert y.shape == (n_seeds, 100)
    assert is_same_interval(x)
    assert np.allclose(y, np.minimum.accumulate(y, axis=-1))

    step = 50
    x, y = get_performance_over_time(cumtimes=cumtimes, perf_vals=perf_vals, step=50)
    assert x.shape == (step,)
    assert y.shape == (n_seeds, step)
    assert is_log_scale(x)
    assert np.allclose(y, np.minimum.accumulate(y, axis=-1))


def test_errors_in_get_performance_over_time():
    n_evals = 1000
    rng = np.random.RandomState(0)
    cumtimes = np.sort(rng.random(size=n_evals))
    perf_vals = rng.random(size=n_evals)
    with pytest.raises(TypeError, match=r"cumtimes and perf_vals must be 2D array*"):
        run_get_performance_over_time(cumtimes, perf_vals)

    cumtimes = np.sort(rng.random(size=(10, n_evals)))
    perf_vals = rng.random(size=(11, n_evals))
    with pytest.raises(ValueError, match=r"The number of seeds used in cumtimes and perf_vals must be identical*"):
        run_get_performance_over_time(cumtimes, perf_vals)

    cumtimes = np.sort(rng.random(size=(1, n_evals)))
    perf_vals = rng.random(size=(1, n_evals + 1))
    with pytest.raises(ValueError, match=r"The shape of each cumtimes and perf_vals for each seed must be identical*"):
        run_get_performance_over_time(cumtimes, perf_vals)


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

    x, y = get_performance_over_time_from_paths(paths, obj_key="loss")
    assert x.shape == (100,)
    assert y.shape == (n_seeds, 100)
    assert is_log_scale(x)
    assert np.allclose(y, np.minimum.accumulate(y, axis=-1))
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
