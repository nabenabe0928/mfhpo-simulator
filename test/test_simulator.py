import multiprocessing
import os
import pytest
import shutil
import unittest
from typing import Any, Dict, Optional, Union

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import CentralWorkerManager, ObjectiveFuncWorker

import numpy as np

import ujson as json


SUBDIR_NAME = "dummy"
PATH = os.path.join(DIR_NAME, SUBDIR_NAME)
DEFAULT_KWARGS = dict(
    subdir_name=SUBDIR_NAME,
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
    continual_max_fidel=10,
    fidel_keys=["epoch"],
)


def dummy_func(
    eval_config: Dict[str, Any],
    fidels: Optional[Dict[str, Union[float, int]]],
    seed: Optional[int],
) -> Dict[str, float]:
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def dummy_no_fidel_func(
    eval_config: Dict[str, Any],
    fidels: Optional[Dict[str, Union[float, int]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    return dict(loss=eval_config["x"], runtime=10)


def dummy_func_with_data(
    eval_config: Dict[str, Any],
    fidels: Optional[Dict[str, Union[float, int]]],
    seed: Optional[int],
    **data_to_scatter: Any,
) -> Dict[str, float]:
    assert len(data_to_scatter) > 0
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def dummy_func_with_many_fidelities(
    eval_config: Dict[str, Any],
    fidels: Optional[Dict[str, Union[float, int]]],
    seed: Optional[int],
    **data_to_scatter: Any,
) -> Dict[str, float]:
    runtime = fidels["z1"] + fidels["z2"] + fidels["z3"]
    return dict(loss=eval_config["x"], runtime=runtime)


def test_error_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    worker = ObjectiveFuncWorker(
        obj_func=dummy_no_fidel_func,
        **kwargs,
    )
    # Objective function did not get keyword `fidels`
    with pytest.raises(ValueError):
        worker(eval_config={"x": 0}, fidels=None)

    shutil.rmtree(worker.dir_name)

    kwargs.pop("continual_max_fidel")
    kwargs.pop("fidel_keys")
    worker = ObjectiveFuncWorker(
        obj_func=dummy_no_fidel_func,
        **kwargs,
    )
    worker(eval_config={"x": 0}, fidels=None)
    # Objective function got keyword `fidels`
    with pytest.raises(ValueError):
        worker(eval_config={"x": 0}, fidels={"epoch": 0})

    shutil.rmtree(worker.dir_name)


def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError):
        ObjectiveFuncWorker(
            obj_func=dummy_no_fidel_func,
            **kwargs,
        )
    if os.path.exists(PATH):
        shutil.rmtree(PATH)


def test_validate_fidel_args():
    kwargs = DEFAULT_KWARGS.copy()
    for fidel_keys in [None, ["a", "b"], []]:
        kwargs["fidel_keys"] = fidel_keys
        with pytest.raises(ValueError):
            ObjectiveFuncWorker(
                obj_func=dummy_no_fidel_func,
                **kwargs,
            )
        if os.path.exists(PATH):
            shutil.rmtree(PATH)


def test_errors_in_proc_output():
    kwargs = DEFAULT_KWARGS.copy()
    # fidels is None or len(fidels.values()) != 1
    with pytest.raises(ValueError):
        worker = ObjectiveFuncWorker(
            obj_func=dummy_func,
            **kwargs,
        )
        worker(eval_config={"x": 1}, fidels={"epoch": 1, "epoch2": 1})

    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    # Fidelity for continual evaluation must be integer
    with pytest.raises(ValueError):
        worker = ObjectiveFuncWorker(
            obj_func=lambda eval_config, fidels, **kwargs: dict(loss=eval_config["x"], runtime=1),
            **kwargs,
        )
        worker(eval_config={"x": 0}, fidels={"epoch": 1.0})

    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    kwargs.pop("continual_max_fidel")
    # The keys in fidels must be identical to fidel_keys
    with pytest.raises(KeyError):
        worker = ObjectiveFuncWorker(
            obj_func=dummy_func,
            **kwargs,
        )
        worker(eval_config={"x": 1}, fidels={"epoch": 1, "epoch2": 1})

    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    kwargs["fidel_keys"] = ["dummy-fidel"]
    # The keys in fidels must be identical to fidel_keys
    with pytest.raises(KeyError):
        worker = ObjectiveFuncWorker(
            obj_func=lambda eval_config, fidels, **kwargs: dict(loss=eval_config["x"], runtime=1),
            **kwargs,
        )
        worker(eval_config={"x": 0}, fidels={"epoch": 1})

    if os.path.exists(PATH):
        shutil.rmtree(PATH)


def test_error_in_keys():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    with pytest.raises(KeyError):
        worker = ObjectiveFuncWorker(
            obj_func=dummy_func,
            obj_keys=["dummy_loss"],
            **kwargs,
        )
        worker(eval_config={"x": 0}, fidels={"epoch": 1})

    shutil.rmtree(worker.dir_name)
    with pytest.raises(KeyError):
        worker = ObjectiveFuncWorker(
            obj_func=dummy_func,
            runtime_key="dummy_runtime",
            **kwargs,
        )
        worker(eval_config={"x": 0}, fidels={"epoch": 1})

    shutil.rmtree(worker.dir_name)

    with pytest.raises(KeyError):
        worker = ObjectiveFuncWorker(
            obj_func=dummy_func,
            obj_keys=["dummy_loss", "loss"],
            **kwargs,
        )
        worker(eval_config={"x": 0}, fidels={"epoch": 1})

    shutil.rmtree(worker.dir_name)


def test_call_with_many_fidelities():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    kwargs["fidel_keys"] = ["z1", "z2", "z3"]
    kwargs.pop("continual_max_fidel")
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func_with_many_fidelities,
        **kwargs,
    )

    for i in range(15):
        results = worker(eval_config={"x": i}, fidels={"z1": i, "z2": i, "z3": i})
        if i >= n_evals:
            assert all(v > 1000 for v in results.values())

    shutil.rmtree(worker.dir_name)


def test_call_with_data():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func_with_data,
        **kwargs,
    )

    data = np.ones(100)
    for i in range(15):
        results = worker(eval_config={"x": i}, fidels={"epoch": i}, data=data)
        if i >= n_evals:
            assert all(v > 1000 for v in results.values())

    shutil.rmtree(worker.dir_name)


def test_call():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func,
        **kwargs,
    )

    assert worker.fidel_keys == ["epoch"]
    assert worker._max_fidel == kwargs["continual_max_fidel"]
    assert worker._runtime_key == "runtime"
    assert worker._obj_keys == ["loss"]

    for i in range(15):
        results = worker(eval_config={"x": i}, fidels={"epoch": i})
        if i >= n_evals:
            assert all(v > 1000 for v in results.values())

    shutil.rmtree(worker.dir_name)


def test_call_considering_state():
    n_evals = 21
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals, n_actual_evals_in_opt=22)
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func,
        **kwargs,
    )
    worker(eval_config={"x": 1}, fidels={"epoch": 10})  # max-fidel and thus no need to cache
    assert len(json.load(open(worker._state_path))) == 0

    for i in range(10):
        for j in range(2):
            last = (i == 9) and (j == 1)
            worker(eval_config={"x": 1}, fidels={"epoch": i + 1})
            states = json.load(open(worker._state_path))
            assert len(states) == int(not last)

            if last:
                continue

            key = next(iter(states))
            ans = 2
            if (i == 0 and j == 0) or (i == 9 and j == 0):
                ans = 1
            assert len(states[key]) == ans

    shutil.rmtree(worker.dir_name)


def remove_tree():
    try:
        shutil.rmtree(PATH)
    except FileNotFoundError:
        pass


def get_n_workers():
    n_workers = 4 if os.system("hostname") == "EB-B9400CBA" else 2  # github actions has only 2 cores
    return n_workers


def test_central_worker_manager():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = get_n_workers()
    kwargs["n_actual_evals_in_opt"] = 15
    manager = CentralWorkerManager(obj_func=dummy_func, **kwargs)
    assert manager.fidel_keys == ["epoch"]
    assert manager._max_fidel == kwargs["continual_max_fidel"]
    assert manager._runtime_key == "runtime"
    assert manager._obj_keys == ["loss"]
    shutil.rmtree(manager.dir_name)


def test_seeds_error_in_central_worker_manager():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 15
    manager = CentralWorkerManager(obj_func=dummy_func, seeds=list(range(n_workers)), **kwargs)
    with pytest.raises(FileExistsError):
        CentralWorkerManager(obj_func=dummy_func, seeds=list(range(n_workers)), **kwargs)

    shutil.rmtree(manager.dir_name)
    with pytest.raises(ValueError):
        CentralWorkerManager(obj_func=dummy_func, seeds=[0], **kwargs)

    remove_tree()


def test_optimize_seq():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    manager = CentralWorkerManager(obj_func=dummy_func, seeds=[0], **kwargs)

    kwargs = dict(
        eval_config={"x": 1},
        fidels={"epoch": 1},
    )
    manager(**kwargs)
    shutil.rmtree(manager.dir_name)


def test_optimize_parallel():
    remove_tree()
    n_workers = get_n_workers()
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_workers"] = n_workers
    kwargs["n_actual_evals_in_opt"] = 16
    manager = CentralWorkerManager(obj_func=dummy_func, seeds=list(range(n_workers)), **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for _ in range(16):
        kwargs = dict(
            eval_config={"x": 1},
            fidels={"epoch": 1},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res.append(r)
    else:
        for r in res:
            r.get()

    pool.close()
    pool.join()

    path = manager.dir_name
    out = json.load(open(os.path.join(path, "results.json")))
    shutil.rmtree(path)
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)


if __name__ == "__main__":
    unittest.main()
