from __future__ import annotations

import os
import pytest
import shutil
import unittest

from benchmark_simulator._constants import AbstractAskTellOptimizer
from benchmark_simulator.simulator import ObjectiveFuncWrapper

import ujson as json

from tests.utils import (
    DIR_PATH,
    SIMPLE_CONFIG,
    SUBDIR_NAME,
    dummy_func,
    dummy_func_with_many_fidelities,
    dummy_no_fidel_func,
    remove_tree,
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


def test_error_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_no_fidel_func,
        **kwargs,
    )
    with pytest.raises(ValueError, match="Objective function did not get keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None)

    shutil.rmtree(worker.dir_name)

    kwargs.pop("continual_max_fidel")
    kwargs.pop("fidel_keys")
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_no_fidel_func,
        **kwargs,
    )
    worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, worker_id=0, fidels=None)  # no error without fidel!
    # Objective function got keyword `fidels`
    with pytest.raises(ValueError, match="Objective function got keyword `fidels`*"):
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 0}, worker_id=0)

    shutil.rmtree(worker.dir_name)


def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError, match="Cannot guarantee that optimziers will not hang"):
        ObjectiveFuncWrapper(
            obj_func=dummy_no_fidel_func,
            **kwargs,
        )
    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)


def test_validate_fidel_args():
    kwargs = DEFAULT_KWARGS.copy()
    for fidel_keys in [None, ["a", "b"], []]:
        kwargs["fidel_keys"] = fidel_keys
        with pytest.raises(ValueError, match="continual_max_fidel is valid only if fidel_keys has only one element*"):
            ObjectiveFuncWrapper(
                obj_func=dummy_no_fidel_func,
                **kwargs,
            )
        if os.path.exists(DIR_PATH):
            shutil.rmtree(DIR_PATH)


def test_errors_in_proc_output():
    kwargs = DEFAULT_KWARGS.copy()
    # fidels is None or len(fidels.values()) != 1
    with pytest.raises(ValueError, match="fidels must have only one element*"):
        worker = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config={"x": 1}, fidels={"epoch": 1, "epoch2": 1}, worker_id=0)

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)

    # Fidelity for continual evaluation must be integer
    with pytest.raises(ValueError, match="Fidelity for continual evaluation must be integer*"):
        worker = ObjectiveFuncWrapper(
            obj_func=lambda eval_config, fidels, **kwargs: dict(loss=eval_config["x"], runtime=1),
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1.0}, worker_id=0)

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)

    with pytest.raises(ValueError, match="Fidelity for continual evaluation must be non-negative*"):
        worker = ObjectiveFuncWrapper(
            obj_func=lambda eval_config, fidels, **kwargs: dict(loss=eval_config["x"], runtime=1),
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": -1}, worker_id=0)

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)

    kwargs.pop("continual_max_fidel")
    # The keys in fidels must be identical to fidel_keys
    with pytest.raises(KeyError, match="The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config={"x": 1}, fidels={"epoch": 1, "epoch2": 1}, worker_id=0)

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)

    kwargs["fidel_keys"] = ["dummy-fidel"]
    # The keys in fidels must be identical to fidel_keys
    with pytest.raises(KeyError, match="The keys in fidels must be identical to fidel_keys*"):
        worker = ObjectiveFuncWrapper(
            obj_func=lambda eval_config, fidels, **kwargs: dict(loss=eval_config["x"], runtime=1),
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0)

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)


def test_error_in_keys():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    with pytest.raises(KeyError, match="The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            obj_keys=["dummy_loss"],
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0)

    shutil.rmtree(worker.dir_name)
    with pytest.raises(KeyError, match="The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            runtime_key="dummy_runtime",
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0)

    shutil.rmtree(worker.dir_name)

    with pytest.raises(KeyError, match="The output of objective must be a superset*"):
        worker = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            obj_keys=["dummy_loss", "loss"],
            **kwargs,
        )
        worker._main_wrapper._proc_obj_func(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}, worker_id=0)

    shutil.rmtree(worker.dir_name)


def test_call_with_many_fidelities():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    kwargs["fidel_keys"] = ["z1", "z2", "z3"]
    kwargs.pop("continual_max_fidel")
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func_with_many_fidelities,
        **kwargs,
    )

    for i in range(15):
        worker._main_wrapper._proc_obj_func(eval_config={"x": i}, fidels={"z1": i, "z2": i, "z3": i}, worker_id=0)

    shutil.rmtree(worker.dir_name)


class _DummyOptConsideringState(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        if self._n_calls == -1:
            # max-fidel and thus no need to cache
            eval_config = {"x": 1}
            fidels = {"epoch": 10}
        else:
            i = self._n_calls // 2
            eval_config = {"x": 1}
            fidels = {"epoch": i + 1}

        self._n_calls += 1
        return eval_config, fidels

    def tell(self, *args, **kwargs):
        pass


class _DummyOpt(AbstractAskTellOptimizer):
    def __init__(self):
        self._n_calls = -1

    def ask(self):
        self._n_calls += 1
        return {"x": self._n_calls}, {"epoch": self._n_calls + 1}

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
        return eval_config, fidels

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


def test_call_considering_state():  # from here
    n_evals = 21
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals, n_actual_evals_in_opt=22)
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        **kwargs,
    )
    opt = _DummyOptConsideringState()
    for k in range(-1, 20):
        if k == -1:
            # max-fidel and thus no need to cache
            eval_config, fidels = worker._main_wrapper._ask_with_timer(opt=opt, worker_id=0)
            worker._main_wrapper._proc_obj_func(eval_config=eval_config, worker_id=0, fidels=fidels)
            worker._main_wrapper._tell_pending_result(opt=opt, worker_id=0)
            assert len(worker._main_wrapper._intermediate_states) == 0
            continue

        i, j = k // 2, k % 2
        last = (i == 9) and (j == 1)
        eval_config, fidels = worker._main_wrapper._ask_with_timer(opt=opt, worker_id=0)
        worker._main_wrapper._proc_obj_func(eval_config=eval_config, worker_id=0, fidels=fidels)
        worker._main_wrapper._tell_pending_result(opt=opt, worker_id=0)

        states = worker._main_wrapper._intermediate_states
        assert len(states) == int(not last)

        if last:
            continue

        key = next(iter(states))
        ans = 2
        if (i == 0 and j == 0) or (i == 9 and j == 0):
            ans = 1
        assert len(states[key]) == ans

    shutil.rmtree(worker.dir_name)


def test_store_config():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = 4
    kwargs["n_actual_evals_in_opt"] = 15
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)
    worker.simulate(_DummyOpt())

    results = json.load(open(os.path.join(worker.dir_name, "results.json")))
    for k in ["seed", "epoch", "x"]:
        assert k in results
        assert len(results[k]) == kwargs["n_evals"]
    shutil.rmtree(worker.dir_name)


def test_store_config_with_conditional():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = 4
    kwargs["n_actual_evals_in_opt"] = 15
    worker = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)
    worker.simulate(_DummyOptCond())
    results = json.load(open(os.path.join(worker.dir_name, "results.json")))
    for k in ["seed", "epoch", "x", "y"]:
        assert k in results
        assert len(results[k]) == kwargs["n_evals"]
    shutil.rmtree(worker.dir_name)


def test_error_in_opt():
    remove_tree()
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
