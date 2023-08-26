from __future__ import annotations

import pytest
import unittest

from benchmark_simulator._constants import AbstractAskTellOptimizer
from benchmark_simulator.simulator import ObjectiveFuncWrapper

import numpy as np

from tests.utils import (
    SIMPLE_CONFIG,
    SUBDIR_NAME,
    cleanup,
    dummy_func,
    dummy_func_with_constant_runtime,
    dummy_func_with_many_fidelities,
    dummy_no_fidel_func,
)


DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    ask_and_tell=True,
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
    continual_max_fidel=10,
    fidel_keys=["epoch"],
)


class _DummyOptConsideringState(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        if self._n_calls == -1:
            # max-fidel and thus no need to cache
            eval_config = SIMPLE_CONFIG
            fidels = {"epoch": 10}
        else:
            i = self._n_calls // 2
            eval_config = SIMPLE_CONFIG
            fidels = {"epoch": i + 1}

        self._n_calls += 1
        return eval_config, fidels, None

    def tell(self, *args, **kwargs):
        pass


class _DummyOpt(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        return {"x": self._n_calls}, {"epoch": self._n_calls + 1}, None

    def tell(self, *args, **kwargs):
        pass


class _DummyOptCond(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        i = self._n_calls
        eval_config = {"x": i} if i < 6 or i % 2 == 0 else {"x": i, "y": i}
        fidels = {"epoch": i + 1}
        return eval_config, fidels, None

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


@cleanup
def test_error_no_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    with pytest.raises(ValueError, match=r"Objective function did not get keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None, config_id=None)


@cleanup
def test_error_unneeded_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    kwargs.pop("fidel_keys")
    worker = ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)
    worker._main_wrapper._proc_obj_func(  # no error without fidel!
        eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None, config_id=None
    )
    # Objective function got keyword `fidels`
    with pytest.raises(ValueError, match=r"Objective function got keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 0}, worker_id=0, config_id=None)


@cleanup
def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError, match=r"Cannot guarantee that optimziers will not hang*"):
        ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)


@cleanup
def test_no_expensive_parallel_sample():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["allow_parallel_sampling"] = True
    kwargs["expensive_sampler"] = True
    with pytest.raises(ValueError, match=r"expensive_sampler and allow_parallel_sampling cannot*"):
        ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)


@cleanup
def _validate_fidel_args(fidel_keys: list[str] | None):
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["fidel_keys"] = fidel_keys
    with pytest.raises(ValueError, match=r"continual_max_fidel is valid only if fidel_keys has only one element*"):
        ObjectiveFuncWrapper(obj_func=dummy_no_fidel_func, **kwargs)


@pytest.mark.parametrize("fidel_keys", (None, ["a", "b"], []))
def test_validate_fidel_args(fidel_keys: list[str] | None):
    _validate_fidel_args(fidel_keys=fidel_keys)


@cleanup
def test_fidel_must_have_only_one_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(ValueError, match=r"fidels must have only one element*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
        worker._main_wrapper._proc_obj_func(
            eval_config=SIMPLE_CONFIG, fidels={"epoch": 1, "epoch2": 1}, worker_id=0, config_id=None
        )


@cleanup
def test_fidel_must_be_int_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(ValueError, match=r"Fidelity for continual evaluation must be integer*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker._main_wrapper._proc_obj_func(
            eval_config=SIMPLE_CONFIG, fidels={"epoch": 1.0}, worker_id=0, config_id=None
        )


@cleanup
def test_fidel_must_be_non_negative_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(ValueError, match=r"Fidelity for continual evaluation must be non-negative*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker._main_wrapper._proc_obj_func(
            eval_config=SIMPLE_CONFIG, fidels={"epoch": -1}, worker_id=0, config_id=None
        )


@cleanup
def test_fidel_keys_must_be_identical_using_weird_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    with pytest.raises(KeyError, match=r"The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
        worker._main_wrapper._proc_obj_func(
            eval_config=SIMPLE_CONFIG, fidels={"epoch": 1, "epoch2": 1}, worker_id=0, config_id=None
        )


@cleanup
def test_fidel_keys_must_be_identical_using_weird_instance():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    kwargs["fidel_keys"] = ["dummy-fidel"]
    with pytest.raises(KeyError, match=r"The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


@cleanup
def _weird_obj_keys(obj_keys: list[str]):
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(KeyError, match=r"The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, obj_keys=["dummy_loss"], **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


@pytest.mark.parametrize("obj_keys", (["dummy_loss"], ["dummy_loss", "loss"]))
def test_weird_obj_keys(obj_keys: list[str]):
    _weird_obj_keys(obj_keys=obj_keys)


@cleanup
def test_weird_runtime_key():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(KeyError, match=r"The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(obj_func=dummy_func, runtime_key="dummy_runtime", **kwargs)
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0, config_id=None)


@cleanup
def test_call_with_many_fidelities():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    kwargs["fidel_keys"] = ["z1", "z2", "z3"]
    kwargs.pop("continual_max_fidel")
    worker = ObjectiveFuncWrapper(obj_func=dummy_func_with_many_fidelities, **kwargs)

    for i in range(15):
        worker._main_wrapper._proc_obj_func(
            eval_config={"x": i}, fidels={"z1": i, "z2": i, "z3": i}, worker_id=0, config_id=None
        )


@cleanup
def test_call_considering_state():
    n_evals = 21
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals, n_actual_evals_in_opt=22)
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
    opt = _DummyOptConsideringState()
    for k in range(-1, 20):
        if k == -1:
            # max-fidel and thus no need to cache
            eval_config, fidels, _ = worker._main_wrapper._ask_with_timer(opt=opt, worker_id=0)
            worker._main_wrapper._proc_obj_func(eval_config=eval_config, worker_id=0, fidels=fidels, config_id=None)
            worker._main_wrapper._tell_pending_result(opt=opt, worker_id=0)
            assert len(worker._main_wrapper._state_tracker._intermediate_states) == 0
            continue

        i, j = k // 2, k % 2
        last = (i == 9) and (j == 1)
        eval_config, fidels, _ = worker._main_wrapper._ask_with_timer(opt=opt, worker_id=0)
        worker._main_wrapper._proc_obj_func(eval_config=eval_config, worker_id=0, fidels=fidels, config_id=None)
        worker._main_wrapper._tell_pending_result(opt=opt, worker_id=0)

        states = worker._main_wrapper._state_tracker._intermediate_states
        assert len(states) == int(not last)

        if last:
            continue

        key = next(iter(states))
        ans = 2
        if (i == 0 and j == 0) or (i == 9 and j == 0):
            ans = 1
        assert len(states[key]) == ans


@cleanup
def _store_actual_cumtime(store_config: bool) -> None:
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_workers=4, n_actual_evals_in_opt=15)
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=store_config, store_actual_cumtime=True, **kwargs)
    worker.simulate(_DummyOpt())

    results = worker.get_results()
    assert "actual_cumtime" in results
    actual_cumtimes = np.array(results["actual_cumtime"])
    assert len(actual_cumtimes) == kwargs["n_evals"]
    assert np.allclose(np.maximum.accumulate(actual_cumtimes), actual_cumtimes)


@pytest.mark.parametrize("store_config", (True, False))
def test_store_actual_cumtime(store_config: bool) -> None:
    _store_actual_cumtime(store_config=store_config)


@cleanup
def _store_config(opt):
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_workers=4, n_actual_evals_in_opt=15)
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)
    worker.simulate(opt())

    results = worker.get_results()
    keys = ["seed", "epoch", "x"]
    if isinstance(opt, _DummyOptCond):
        keys.append("y")

    for k in keys:
        assert k in results
        assert len(results[k]) == kwargs["n_evals"]


@pytest.mark.parametrize("opt", (_DummyOpt, _DummyOptCond))
def test_store_config(opt):
    _store_config(opt=opt)


@cleanup
def test_error_in_opt():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutAsk())
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutTell())
    with pytest.raises(ValueError, match=r"opt must have `ask` and `tell`"):
        worker.simulate(_DummyWithoutAnything())


if __name__ == "__main__":
    unittest.main()
