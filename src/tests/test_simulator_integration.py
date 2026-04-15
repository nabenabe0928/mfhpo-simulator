from __future__ import annotations

import unittest

import numpy as np

from src._constants import AbstractAskTellOptimizer
from src.simulator import ObjectiveFuncWrapper
from src.tests.utils import dummy_no_fidel_func
from src.tests.utils import simplest_dummy_func


DEFAULT_KWARGS = dict(
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
)


class _DummyOpt(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        return {"x": self._n_calls}

    def tell(self, *args, **kwargs):
        pass


# --- get_optimizer_overhead tests ---


def test_get_optimizer_overhead():
    """get_optimizer_overhead should return sampling time data with correct structure."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    wrapper.simulate(_DummyOpt())

    overhead = wrapper.get_optimizer_overhead()
    assert "before_sample" in overhead
    assert "after_sample" in overhead
    assert "worker_index" in overhead
    # n_evals + n_workers - 1 ask() calls
    expected_len = kwargs["n_evals"] + kwargs["n_workers"] - 1
    assert len(overhead["before_sample"]) == expected_len
    assert len(overhead["after_sample"]) == expected_len
    assert len(overhead["worker_index"]) == expected_len
    # after_sample >= before_sample
    for before, after in zip(overhead["before_sample"], overhead["after_sample"]):
        assert after >= before


# --- property tests ---


def test_wrapper_properties():
    """Public properties should return expected values."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)

    assert wrapper.n_workers == kwargs["n_workers"]
    assert wrapper.n_actual_evals_in_opt == kwargs["n_actual_evals_in_opt"]
    assert wrapper.obj_keys == ["loss"]
    assert wrapper.runtime_key == "runtime"


# --- result ordering with expensive_sampler=True ---


class _MultiWorkerOpt(AbstractAskTellOptimizer):
    """Optimizer that assigns different runtimes via eval_config to create out-of-order completions."""

    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        return {"x": self._n_calls}

    def tell(self, *args, **kwargs):
        pass


def test_results_sorted_by_cumtime_with_expensive_sampler():
    """With expensive_sampler=True, get_results() should have cumtimes in non-decreasing order."""
    n_evals = 8
    wrapper = ObjectiveFuncWrapper(
        obj_func=simplest_dummy_func,
        n_workers=4,
        n_actual_evals_in_opt=n_evals + 5,
        n_evals=n_evals,
        expensive_sampler=True,
    )
    wrapper.simulate(_MultiWorkerOpt())
    results = wrapper.get_results()
    cumtimes = np.array(results["cumtime"])
    # Results should be sorted by cumtime
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


def test_results_cumtime_monotonic_without_expensive_sampler():
    """Without expensive_sampler, get_results() cumtimes should also be non-decreasing."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    wrapper.simulate(_DummyOpt())
    results = wrapper.get_results()
    cumtimes = np.array(results["cumtime"])
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


# --- tell_pending_result edge cases ---


def test_tell_skips_none_pending_results():
    """_tell_pending_result should skip workers with None pending results."""
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_workers=2, n_actual_evals_in_opt=15)
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    wrapper.simulate(_DummyOpt())
    results = wrapper.get_results()
    # Should complete without errors and have the expected number of results
    assert len(results["cumtime"]) == kwargs["n_evals"]


def test_multi_worker_all_results_collected():
    """With multiple workers, all n_evals results should be collected."""
    n_workers = 4
    n_evals = 12
    wrapper = ObjectiveFuncWrapper(
        obj_func=dummy_no_fidel_func,
        n_workers=n_workers,
        n_actual_evals_in_opt=n_evals + n_workers + 1,
        n_evals=n_evals,
    )
    wrapper.simulate(_DummyOpt())
    results = wrapper.get_results()
    assert len(results["cumtime"]) == n_evals
    assert len(results["loss"]) == n_evals
    assert len(results["worker_index"]) == n_evals


if __name__ == "__main__":
    unittest.main()
