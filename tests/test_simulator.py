from __future__ import annotations

import multiprocessing
import os
import pytest
import shutil
import time
import unittest
from typing import Any

from benchmark_simulator._constants import _SharedDataFileNames, _TIME_VALUES
from benchmark_simulator.simulator import ObjectiveFuncWrapper, get_multiple_wrappers

import numpy as np

import ujson as json

from tests.utils import (
    DIR_PATH,
    ON_UBUNTU,
    SIMPLE_CONFIG,
    SUBDIR_NAME,
    cleanup,
    dummy_func,
    dummy_func_with_constant_runtime,
    dummy_func_with_many_fidelities,
    dummy_no_fidel_func,
    get_n_workers,
    get_worker_wrapper,
    remove_tree,
)


DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
    continual_max_fidel=10,
    fidel_keys=["epoch"],
)


def dummy_func_with_data(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
    **data_to_scatter: Any,
) -> dict[str, float]:
    assert len(data_to_scatter) > 0
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


@cleanup
def test_error_no_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    worker = get_worker_wrapper(obj_func=dummy_no_fidel_func, **kwargs)
    # Objective function did not get keyword `fidels`
    with pytest.raises(ValueError, match="Objective function did not get keyword `fidels`*"):
        worker(eval_config=SIMPLE_CONFIG, fidels=None)


@cleanup
def test_error_unneeded_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    kwargs.pop("fidel_keys")
    worker = get_worker_wrapper(obj_func=dummy_no_fidel_func, **kwargs)
    worker(eval_config=SIMPLE_CONFIG)  # no error without fidel!
    # Objective function got keyword `fidels`
    with pytest.raises(ValueError, match="Objective function got keyword `fidels`*"):
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 0})


@cleanup
def test_get_multiple_wrappers():
    wrappers = get_multiple_wrappers(obj_func=dummy_func, n_workers=2, save_dir_name=SUBDIR_NAME)
    assert all(isinstance(w, ObjectiveFuncWrapper) for w in wrappers)
    assert len(wrappers) == 2
    with pytest.raises(FileExistsError):
        wrappers = get_multiple_wrappers(obj_func=dummy_func, n_workers=2, save_dir_name=SUBDIR_NAME)


@cleanup
def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError, match="Cannot guarantee that optimziers will not hang"):
        get_worker_wrapper(obj_func=dummy_no_fidel_func, **kwargs)


@cleanup
def _validate_fidel_args(fidel_keys: list[str] | None):
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["fidel_keys"] = fidel_keys
    with pytest.raises(ValueError, match="continual_max_fidel is valid only if fidel_keys has only one element*"):
        get_worker_wrapper(obj_func=dummy_no_fidel_func, **kwargs)


@pytest.mark.parametrize("fidel_keys", (None, ["a", "b"], []))
def test_validate_fidel_args(fidel_keys: list[str] | None):
    _validate_fidel_args(fidel_keys=fidel_keys)


@cleanup
def test_fidel_must_have_only_one_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    # fidels is None or len(fidels.values()) != 1
    with pytest.raises(ValueError, match="fidels must have only one element*"):
        worker = get_worker_wrapper(obj_func=dummy_func, **kwargs)
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1, "epoch2": 1})


@cleanup
def test_fidel_must_be_int_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(ValueError, match="Fidelity for continual evaluation must be integer*"):
        worker = get_worker_wrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1.0})


@cleanup
def test_fidel_must_be_non_negative_for_continual():
    kwargs = DEFAULT_KWARGS.copy()
    with pytest.raises(ValueError, match="Fidelity for continual evaluation must be non-negative*"):
        worker = get_worker_wrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": -1})


@cleanup
def test_fidel_keys_must_be_identical_using_weird_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    with pytest.raises(KeyError, match="The keys in fidels must be identical to fidel_keys*"):
        worker = get_worker_wrapper(obj_func=dummy_func, **kwargs)
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1, "epoch2": 1})


@cleanup
def test_fidel_keys_must_be_identical_using_weird_instance():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("continual_max_fidel")
    kwargs["fidel_keys"] = ["dummy-fidel"]
    with pytest.raises(KeyError, match="The keys in fidels must be identical to fidel_keys*"):
        worker = get_worker_wrapper(obj_func=dummy_func_with_constant_runtime, **kwargs)
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1})


@cleanup
def _weird_obj_keys(obj_keys: list[str]):
    kwargs = DEFAULT_KWARGS.copy()
    worker = get_worker_wrapper(obj_func=dummy_func, obj_keys=obj_keys, **kwargs)
    assert worker.obj_keys == obj_keys
    with pytest.raises(KeyError, match="The output of objective must be a superset*"):
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1})


@pytest.mark.parametrize("obj_keys", (["dummy_loss"], ["dummy_loss", "loss"]))
def test_weird_obj_keys(obj_keys: list[str]):
    _weird_obj_keys(obj_keys=obj_keys)


@cleanup
def test_weird_runtime_key():
    kwargs = DEFAULT_KWARGS.copy()
    worker = get_worker_wrapper(obj_func=dummy_func, runtime_key="dummy_runtime", **kwargs)
    assert worker.runtime_key == "dummy_runtime"
    with pytest.raises(KeyError, match="The output of objective must be a superset*"):
        worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1})


def _check_call(worker: ObjectiveFuncWrapper, fidel_keys: list[str], n_evals: int, data=None):
    for i in range(15):
        if data is not None:
            results = worker(eval_config={"x": i}, fidels={k: i for k in fidel_keys}, data=data)
        else:
            results = worker(eval_config={"x": i}, fidels={k: i for k in fidel_keys})
        if i >= n_evals:  # finish --> should be inf!
            assert all(v > 1000 for v in results.values())


@cleanup
def test_call_with_many_fidelities():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    kwargs["fidel_keys"] = ["z1", "z2", "z3"]
    kwargs.pop("continual_max_fidel")
    worker = get_worker_wrapper(obj_func=dummy_func_with_many_fidelities, **kwargs)
    assert worker.fidel_keys == ["z1", "z2", "z3"]
    _check_call(worker=worker, fidel_keys=kwargs["fidel_keys"], n_evals=n_evals)


@cleanup
def test_call_with_data():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    worker = get_worker_wrapper(obj_func=dummy_func_with_data, **kwargs)
    _check_call(worker=worker, fidel_keys=kwargs["fidel_keys"], n_evals=n_evals, data=np.ones(100))


@cleanup
def test_call():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    worker = get_worker_wrapper(obj_func=dummy_func, **kwargs)
    _check_call(worker=worker, fidel_keys=kwargs["fidel_keys"], n_evals=n_evals)


def test_call_considering_state():
    n_evals = 21
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals, n_actual_evals_in_opt=22)
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        launch_multiple_wrappers_from_user_side=True,
        **kwargs,
    )
    worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": 10})  # max-fidel and thus no need to cache
    assert len(json.load(open(worker._main_wrapper._paths.state_cache))) == 0

    for i in range(10):
        for j in range(2):
            last = (i == 9) and (j == 1)
            worker(eval_config=SIMPLE_CONFIG, fidels={"epoch": i + 1})
            states = json.load(open(worker._main_wrapper._paths.state_cache))
            assert len(states) == int(not last)

            if last:
                continue

            key = next(iter(states))
            ans = 2
            if (i == 0 and j == 0) or (i == 9 and j == 0):
                ans = 1
            assert len(states[key]) == ans

    shutil.rmtree(worker.dir_name)


def test_central_worker_manager():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = get_n_workers()
    kwargs["n_actual_evals_in_opt"] = 15
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)
    assert manager.fidel_keys == ["epoch"]
    assert manager.runtime_key == "runtime"
    assert manager.obj_keys == ["loss"]
    shutil.rmtree(manager.dir_name)


def test_store_config():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func, launch_multiple_wrappers_from_user_side=True, store_config=True, **kwargs
    )
    worker(**dict(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}))
    shutil.rmtree(worker.dir_name)

    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 15
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for i in range(15):
        kwargs = dict(
            eval_config={"x": i},
            fidels={"epoch": i + 1},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res.append(r)
    else:
        for r in res:
            r.get()

    pool.close()
    pool.join()

    results = json.load(open(os.path.join(manager.dir_name, "results.json")))
    for k in ["seed", "epoch", "x"]:
        assert k in results
        assert len(results[k]) == len(results["loss"])
    shutil.rmtree(manager.dir_name)


def test_store_config_with_conditional():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func, launch_multiple_wrappers_from_user_side=True, store_config=True, **kwargs
    )
    worker(**dict(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}))
    worker(**dict(eval_config={"x": 1, "y": 2}, fidels={"epoch": 1}))
    shutil.rmtree(worker.dir_name)

    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 15
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, store_config=True, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for i in range(15):
        kwargs = dict(
            eval_config={"x": i} if i < 6 or i % 2 == 0 else {"x": i, "y": i},
            fidels={"epoch": i + 1},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res.append(r)
    else:
        for r in res:
            r.get()

    pool.close()
    pool.join()

    results = json.load(open(os.path.join(manager.dir_name, "results.json")))
    for k in ["seed", "epoch", "x", "y"]:
        assert k in results
        assert len(results[k]) == len(results["loss"])
    shutil.rmtree(manager.dir_name)


def test_init_alloc_without_error():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, **kwargs)

    for i in range(10):
        kwargs = dict(
            eval_config={"x": i},
            fidels={"epoch": i + 1},
        )
        manager(**kwargs)
    else:
        manager._pid_to_index = {}
        manager(**kwargs)

    shutil.rmtree(manager.dir_name)


def test_interrupted():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWrapper(
        obj_func=dummy_func, launch_multiple_wrappers_from_user_side=True, store_config=True, **kwargs
    )
    data = json.load(open(worker._main_wrapper._paths.worker_cumtime))
    with open(worker._main_wrapper._paths.worker_cumtime, mode="w") as f:
        data[worker._main_wrapper._worker_vars.worker_id] = _TIME_VALUES.crashed
        json.dump(data, f)

    worker(**dict(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}))  # Nothing happens for init
    with pytest.raises(InterruptedError):
        worker(**dict(eval_config=SIMPLE_CONFIG, fidels={"epoch": 1}))

    shutil.rmtree(worker.dir_name)


def test_seed_error_in_central_worker_manager():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 15
    ObjectiveFuncWrapper(obj_func=dummy_func, seed=0, **kwargs)
    with pytest.raises(FileExistsError):
        ObjectiveFuncWrapper(obj_func=dummy_func, seed=0, **kwargs)

    remove_tree()


def test_init_alloc_in_central_worker_manager():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = 1
    kwargs["n_actual_evals_in_opt"] = 15
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, seed=0, **kwargs)
    kwargs = dict(
        eval_config=SIMPLE_CONFIG,
        fidels={"epoch": 1},
    )
    for _ in range(2):
        manager(**kwargs)

    remove_tree()


def test_optimize_seq():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, seed=0, **kwargs)
    n_evals = kwargs["n_evals"]
    n_actual_evals = 15

    kwargs = dict(
        eval_config=SIMPLE_CONFIG,
        fidels={"epoch": 1},
    )
    for _ in range(n_actual_evals):
        manager(**kwargs)

    path = manager.dir_name
    out = json.load(open(os.path.join(path, "results.json")))
    assert len(out["cumtime"]) >= n_evals
    shutil.rmtree(manager.dir_name)


def run_optimize_parallel(wrong_n_workers: bool, remained_file: bool = False):
    remove_tree()
    n_workers = get_n_workers()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 16
    manager = ObjectiveFuncWrapper(obj_func=dummy_func, seed=0, **kwargs)

    if remained_file:
        with open(os.path.join(DIR_PATH, _SharedDataFileNames.proc_alloc.value), mode="w") as f:
            json.dump({"0": 0, "1": 1, "2": 2}, f)

    pool = multiprocessing.Pool(processes=n_workers + wrong_n_workers)
    try:
        res = []
        for _ in range(16):
            time.sleep(0.01)
            kwargs = dict(
                eval_config=SIMPLE_CONFIG,
                fidels={"epoch": 1},
            )
            r = pool.apply_async(manager, kwds=kwargs)
            res.append(r)
        else:
            for r in res:
                r.get()
    except Exception as e:
        pool.close()
        raise e
    else:
        pool.close()
        pool.join()

    path = manager.dir_name
    out = json.load(open(os.path.join(path, "results.json")))
    shutil.rmtree(path)
    diffs = np.abs(out["cumtime"] - np.maximum.accumulate(out["cumtime"]))
    assert np.allclose(diffs, 0.0)


@pytest.mark.parametrize("wrong_n_workers", (True, False))
def test_optimize_parallel(wrong_n_workers: bool):
    if not ON_UBUNTU and wrong_n_workers:
        # TODO: It SOMETIMES (but not always) hangs on MacOS, but I am not sure if that is because of Mac or pytest
        return

    if wrong_n_workers:
        with pytest.raises(ProcessLookupError):
            run_optimize_parallel(wrong_n_workers)
        # TODO: Somehow the following lines hang when using covtest
        # with pytest.raises(ValueError, match=r"Timeout in the allocation of procs*"):
        #     run_optimize_parallel(wrong_n_workers, remained_file=True)
    else:
        run_optimize_parallel(wrong_n_workers)


if __name__ == "__main__":
    unittest.main()
