from __future__ import annotations

import multiprocessing
import os
import pytest
import shutil
import time
import unittest
from typing import Any

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator._secure_proc import _wait_until_next
from benchmark_simulator.simulator import CentralWorkerManager

import ujson as json


SUBDIR_NAME = "dummy"
PATH = os.path.join(DIR_NAME, SUBDIR_NAME)
DEFAULT_KWARGS = dict(
    subdir_name=SUBDIR_NAME,
    n_workers=4,
    n_actual_evals_in_opt=12,
    n_evals=8,
    continual_max_fidel=10,
    fidel_keys=["epoch"],
)


def remove_tree():
    try:
        shutil.rmtree(PATH)
    except FileNotFoundError:
        pass


def get_n_workers():
    n_workers = 4 if os.uname().nodename == "EB-B9400CBA" else 2  # github actions has only 2 cores
    return n_workers


def dummy_func_with_wait(
    eval_config: dict[str, Any],
    fidels: dict[str, int | float] | None,
    seed: int | None,
) -> dict[str, float]:
    time.sleep(1.0)
    return dict(loss=eval_config["x"], runtime=fidels["epoch"])


def test_timeout_error_in_wait_until_next():
    file_name = "test/dummy_cumtime.json"
    with open(file_name, mode="w") as f:
        json.dump({"a": 1.5, "b": 1.0}, f, indent=4)

    with pytest.raises(TimeoutError, match="The simulation was terminated due to too long waiting time*"):
        _wait_until_next(path=file_name, worker_id="a", warning_interval=2, max_waiting_time=2.2)

    os.remove(file_name)


def test_timeout_error_by_wait():
    remove_tree()
    kwargs = DEFAULT_KWARGS.copy()
    n_workers = get_n_workers()
    kwargs["n_workers"] = n_workers
    manager = CentralWorkerManager(obj_func=dummy_func_with_wait, max_waiting_time=0.5, **kwargs)

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


if __name__ == "__main__":
    unittest.main()
