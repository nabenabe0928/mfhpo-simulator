from __future__ import annotations

import unittest

import numpy as np
import pytest

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
        return {"x": self._n_calls}, None

    def tell(self, *args, **kwargs):
        pass


class _DummyOptWithConfigId(AbstractAskTellOptimizer):
    """Optimizer that returns config_id for config tracking tests."""

    def __init__(self, valid: bool = True):
        self._n_calls = -1
        self._valid = valid

    def ask(self):
        self._n_calls += 1
        if self._valid:
            # Same config_id=0 always maps to same config {"x": 0}
            config_id = 0 if self._n_calls % 2 == 0 else 1
            return {"x": config_id}, config_id
        else:
            # Invalid: config_id=0 with different configs
            return {"x": self._n_calls}, 0

    def tell(self, *args, **kwargs):
        pass


class _DummyOptWithConfigIdNoFidel(AbstractAskTellOptimizer):
    """Optimizer that returns config_id."""

    def __init__(self, valid: bool = True):
        self._n_calls = -1
        self._valid = valid

    def ask(self):
        self._n_calls += 1
        if self._valid:
            config_id = self._n_calls % 3
            return {"x": config_id}, config_id
        else:
            # config_id=0 with different configs
            return {"x": self._n_calls}, 0

    def tell(self, *args, **kwargs):
        pass


# --- config_tracking=False tests ---


def test_config_tracking_disabled_skips_validation():
    """When config_tracking=False, duplicated config_id with different configs should NOT raise."""
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["config_tracking"] = False
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    # This optimizer gives config_id=0 to different configs — would fail with config_tracking=True
    wrapper.simulate(_DummyOptWithConfigId(valid=False))
    results = wrapper.get_results()
    assert len(results["cumtime"]) == kwargs["n_evals"]


def test_config_tracking_enabled_catches_invalid():
    """When config_tracking=True (default), duplicated config_id with different configs should raise."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    with pytest.raises(ValueError, match=r".*got the duplicated config_id.*"):
        wrapper.simulate(_DummyOptWithConfigId(valid=False))


def test_config_tracking_enabled_valid():
    """When config_tracking=True, consistent config_id should pass without errors."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    wrapper.simulate(_DummyOptWithConfigId(valid=True))
    results = wrapper.get_results()
    assert len(results["cumtime"]) == kwargs["n_evals"]


def test_config_tracking_valid():
    """Config tracking works correctly."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, config_tracking=True, **kwargs)
    wrapper.simulate(_DummyOptWithConfigIdNoFidel(valid=True))
    results = wrapper.get_results()
    assert len(results["cumtime"]) == kwargs["n_evals"]


def test_config_tracking_invalid():
    """Config tracking catches duplicated config_id."""
    kwargs = DEFAULT_KWARGS.copy()
    wrapper = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, config_tracking=True, **kwargs)
    with pytest.raises(ValueError, match=r".*got the duplicated config_id.*"):
        wrapper.simulate(_DummyOptWithConfigIdNoFidel(valid=False))


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
        return {"x": self._n_calls}, None

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
