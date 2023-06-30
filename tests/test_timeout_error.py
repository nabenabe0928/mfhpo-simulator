from __future__ import annotations

import multiprocessing
import os
import pytest
import shutil
import sys
import time
import unittest
from typing import Any

from benchmark_simulator._secure_proc import _wait_until_next
from benchmark_simulator._utils import _SecureLock
from benchmark_simulator.simulator import ObjectiveFuncWrapper

import ujson as json

from tests.utils import SUBDIR_NAME, get_n_workers, remove_tree


DEFAULT_KWARGS = dict(
    save_dir_name=SUBDIR_NAME,
    n_workers=4,
    n_actual_evals_in_opt=12,
    n_evals=8,
    continual_max_fidel=10,
    fidel_keys=["epoch"],
)


def dummy_func_with_wait(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    time.sleep(1.0)
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def dummy_func_with_crash(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    if eval_config["x"] == 0:
        sys.exit()

    time.sleep(0.1)
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def test_timeout_error_in_wait_until_next():
    file_name = "test/dummy_cumtime.json"
    with open(file_name, mode="w") as f:
        json.dump({"a": 1.5, "b": 1.0}, f, indent=4)

    lock = _SecureLock()
    with pytest.raises(TimeoutError, match="The simulation was terminated due to too long waiting time*"):
        _wait_until_next(
            path=file_name, worker_id="a", warning_interval=2, max_waiting_time=2.2, waiting_time=1e-4, lock=lock
        )

    os.remove(file_name)


def test_timeout_error_by_wait():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    manager = ObjectiveFuncWrapper(obj_func=dummy_func_with_wait, max_waiting_time=0.5, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = []
    for _ in range(12):
        kwargs = dict(
            eval_config={"x": 1},
            fidels={"epoch": 1},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res.append(r)
    else:
        n_timeout = 0
        n_interrupted = 0
        for i, r in enumerate(res):
            try:
                r.get()
            except TimeoutError:
                n_timeout += 1
            except InterruptedError:
                n_interrupted += 1
            except Exception as e:
                raise RuntimeError(f"The first {n_workers} run must be timeout, but the {i+1}-th run failed with {e}")

        assert n_workers == n_timeout + 1

        # It should happen, but we do not know how many times it happens
        assert n_interrupted > 5

    pool.close()
    pool.join()

    shutil.rmtree(manager.dir_name)


def test_timeout_error_by_duplicated_worker():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    manager = ObjectiveFuncWrapper(obj_func=dummy_func_with_crash, max_waiting_time=0.5, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = {}
    for i in range(15):
        kwargs = dict(
            eval_config={"x": i},
            fidels={"epoch": i},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res[i] = r
    else:
        n_timeout = 0
        n_no_proc = 0
        for i in range(1, n_workers + 3):
            if i < n_workers + 3:
                try:
                    res[i].get()
                except TimeoutError:
                    n_timeout += 1
                except ProcessLookupError:
                    n_no_proc += 1
                except Exception as e:
                    raise RuntimeError(
                        f"The first {n_workers} run must be timeout, but the {i+1}-th run failed with {e}"
                    )
            else:
                with pytest.raises(InterruptedError):
                    res[i].get()

        assert n_workers == n_timeout + 1
        assert n_no_proc > 0

    pool.close()
    # pool.join() <== we do not call it because one of the workers is terminated and the info cannot be joined.
    shutil.rmtree(manager.dir_name)


def dummy_func_with_pseudo_crash(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    if eval_config["x"] == 0:
        time.sleep(2)

    time.sleep(0.01)
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def test_timeout_error_by_pseudo_crash():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    manager = ObjectiveFuncWrapper(obj_func=dummy_func_with_pseudo_crash, max_waiting_time=0.5, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = {}
    for i in range(15):
        kwargs = dict(
            eval_config={"x": i},
            fidels={"epoch": i},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res[i] = r

    for i in range(n_workers):
        try:
            res[i].get()
        except TimeoutError:
            pass
        except Exception as e:
            raise RuntimeError(f"The first {n_workers} run must be timeout, but the {i+1}-th run failed with {e}")
    for i in range(n_workers, 15):
        with pytest.raises(InterruptedError):
            res[i].get()

    pool.close()
    pool.join()
    shutil.rmtree(manager.dir_name)


def test_timeout_error_by_pseudo_crash_at_intermidiate():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    manager = ObjectiveFuncWrapper(obj_func=dummy_func_with_pseudo_crash, max_waiting_time=0.5, **kwargs)

    pool = multiprocessing.Pool(processes=n_workers)
    res = {}
    # 1 (ok)      5 (timeout)
    # 2 (ok)      6 (timeout)
    # 3 (ok)      7 (dead, but with return as it is a pseudo crash)
    # 4 (timeout) 8 (should be interrupted)
    for i in range(15):
        kwargs = dict(
            eval_config={"x": i - n_workers - 2},
            fidels={"epoch": i + 1},
        )
        r = pool.apply_async(manager, kwds=kwargs)
        res[i] = r

    for i in range(3):
        res[i].get()
    for i in range(3, n_workers + 2):
        with pytest.raises(TimeoutError):
            res[i].get()

    res[n_workers + 2].get()  # no error
    for i in range(n_workers + 3, 15):
        with pytest.raises(InterruptedError):
            res[i].get()

    pool.close()
    pool.join()
    shutil.rmtree(manager.dir_name)


if __name__ == "__main__":
    unittest.main()
