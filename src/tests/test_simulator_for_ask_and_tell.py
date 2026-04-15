from __future__ import annotations

import unittest

import numpy as np
import pytest

from src._constants import AbstractAskTellOptimizer
from src.simulator import ObjectiveFuncWrapper
from src.tests.utils import dummy_func
from src.tests.utils import dummy_func_with_constant_runtime
from src.tests.utils import dummy_func_with_many_fidelities
from src.tests.utils import dummy_no_fidel_func
from src.tests.utils import SIMPLE_CONFIG


DEFAULT_KWARGS = dict(
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
    fidel_keys=["epoch"],
)


class _DummyOpt(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        return {"x": self._n_calls}, {"epoch": self._n_calls + 1}, None

    def tell(self, *args, **kwargs):
        pass


class _DummyWithoutAsk:
    def tell(self, *args, **kwargs):
        pass


class _DummyWithoutTell:
    def ask(self, *args, **kwargs):
        pass


class _DummyWithoutAnything:
    pass


def test_error_no_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    with pytest.raises(ValueError, match=r"Objective function did not get keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None, config_id=None)


def test_error_unneeded_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("fidel_keys")
    worker = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    worker._main_wrapper._proc_obj_func(  # no error without fidel!
        eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None, config_id=None
    )
    # Objective function got keyword `fidels`
    with pytest.raises(ValueError, match=r"Objective function got keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 0}, worker_id=0, config_id=None)


def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError, match=r"Cannot guarantee that optimziers will not hang*"):
        ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)


def test_no_expensive_parallel_sample():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["allow_parallel_sampling"] = True
    kwargs["expensive_sampler"] = True
    with pytest.raises(ValueError, match=r"expensive_sampler and allow_parallel_sampling cannot*"):
        ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)


def test_fidel_keys_must_be_identical_using_weird_call():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(KeyError, match=r"The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
        worker._main_wrapper._proc_obj_func(
            eval_config=SIMPLE_CONFIG, fidels={"epoch": 1, "epoch2": 1}, worker_id=0, config_id=None
        )


def test_fidel_keys_must_be_identical_using_weird_instance():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["fidel_keys"] = ["dummy-fidel"]
    with pytest.raises(KeyError, match=r"The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


def _weird_obj_keys(obj_keys: list[str]):
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(KeyError, match=r"The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, obj_keys=["dummy_loss"], **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


@pytest.mark.parametrize("obj_keys", (["dummy_loss"], ["dummy_loss", "loss"]))
def test_weird_obj_keys(obj_keys: list[str]):
    _weird_obj_keys(obj_keys=obj_keys)


def test_weird_runtime_key():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(KeyError, match=r"The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, runtime_key="dummy_runtime", **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


def test_call_with_many_fidelities():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    kwargs["fidel_keys"] = ["z1", "z2", "z3"]
    worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_many_fidelities, **kwargs)

    for i in range(15):
        worker._main_wrapper._proc_obj_func(
            eval_config={"x": i}, fidels={"z1": i, "z2": i, "z3": i}, worker_id=0, config_id=None
        )


def test_store_actual_cumtime() -> None:
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_workers=4, n_actual_evals_in_opt=15)
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_actual_cumtime=True, **kwargs)
    worker.simulate(_DummyOpt())

    results = worker.get_results()
    assert "actual_cumtime" in results
    actual_cumtimes = np.array(results["actual_cumtime"])
    assert len(actual_cumtimes) == kwargs["n_evals"]
    if np.size(actual_cumtimes) != 0:
        assert np.allclose(np.maximum.accumulate(actual_cumtimes), actual_cumtimes)


def test_error_in_opt():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutAsk())
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutTell())
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutAnything())


if __name__ == "__main__":
    unittest.main()
