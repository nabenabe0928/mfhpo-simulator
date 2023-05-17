import os
import pytest
import shutil
import unittest
from typing import Any, Dict, Optional

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import ObjectiveFuncWorker

import ujson as json


SUBDIR_NAME = "dummy"
DEFAULT_KWARGS = dict(
    subdir_name=SUBDIR_NAME,
    n_workers=1,
    n_actual_evals_in_opt=11,
    n_evals=10,
    max_fidel=10,
)


def dummy_func(
    eval_config: Dict[str, Any],
    fidel: Optional[int],
    seed: Optional[int],
) -> Dict[str, float]:
    return dict(loss=eval_config["x"], runtime=fidel)


def dummy_no_fidel_func(
    eval_config: Dict[str, Any],
    fidel: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    return dict(loss=eval_config["x"], runtime=10)


def dummy_func_with_data(
    eval_config: Dict[str, Any],
    fidel: Optional[int],
    seed: Optional[int],
    **data_to_scatter: Any,
) -> Dict[str, float]:
    return dict(loss=eval_config["x"], runtime=fidel)


def test_error_fidel_in_call():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.pop("max_fidel")
    worker = ObjectiveFuncWorker(
        obj_func=dummy_no_fidel_func,
        **kwargs,
    )
    worker(eval_config={"x": 0}, fidel=None)
    with pytest.raises(ValueError):
        worker(eval_config={"x": 0}, fidel=0)

    shutil.rmtree(worker.dir_name)


def test_guarantee_no_hang():
    kwargs = DEFAULT_KWARGS.copy()
    kwargs["n_actual_evals_in_opt"] = 10
    with pytest.raises(ValueError):
        ObjectiveFuncWorker(
            obj_func=dummy_no_fidel_func,
            **kwargs,
        )
    path = os.path.join(DIR_NAME, SUBDIR_NAME)
    if os.path.exists(path):
        shutil.rmtree(path)


def test_call():
    n_evals = 10
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals)
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func,
        **kwargs,
    )
    for i in range(15):
        results = worker(eval_config={"x": i}, fidel=i)
        if i >= n_evals:
            assert all(v > 1000 for v in results.values())

    shutil.rmtree(worker.dir_name)


def test_call_considering_state():
    n_evals = 20
    kwargs = DEFAULT_KWARGS.copy()
    kwargs.update(n_evals=n_evals, n_actual_evals_in_opt=21)
    worker = ObjectiveFuncWorker(
        obj_func=dummy_func,
        **kwargs,
    )
    for i in range(10):
        for j in range(2):
            last = (i == 9) and (j == 1)
            worker(eval_config={"x": 1}, fidel=i + 1)
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


if __name__ == "__main__":
    unittest.main()
