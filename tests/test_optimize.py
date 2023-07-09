from __future__ import annotations

import os
import pytest
import shutil
import unittest
from typing import Any

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import ObjectiveFuncWrapper

import ConfigSpace as CS

from dehb import DEHB

import numpy as np

from tests.utils import IS_LOCAL


class ToyFunc:
    def __call__(
        self, eval_config: dict[str, float], fidels: dict[str, int | float] | None, seed: int | None = None
    ) -> dict[str, float]:
        return dict(loss=eval_config["x"] ** 2, runtime=fidels["epoch"] / 1.0)

    @property
    def config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5.0, 5.0))
        return cs

    @property
    def min_fidel(self):
        return 1

    @property
    def max_fidel(self):
        return 9


class Wrapper:
    def __init__(self, bench: Any):
        self._bench = bench

    def __call__(
        self,
        eval_config: dict[str, Any],
        fidels: dict[str, int | float],
        seed: int | None,
        **data_to_scatter: Any,
    ) -> dict[str, Any]:
        output = self._bench(eval_config, fidels, seed, **data_to_scatter)
        ret_vals = dict(fitness=output["loss"], cost=output["runtime"])
        return ret_vals


class DEHBObjectiveFuncWrapper(ObjectiveFuncWrapper):
    def __call__(self, config: dict[str, Any], budget: int, **data_to_scatter: Any) -> dict[str, float]:
        return super().__call__(eval_config=config, fidels={"epoch": int(budget)})


def run_dehb(n_workers: int, max_total_eval_time: float):
    n_actual_evals_in_opt = 100 + n_workers
    save_dir_name = "dummy"
    log_file_name = "dehb-log/"
    path = os.path.join(DIR_NAME, save_dir_name)
    obj_func = ToyFunc()

    if os.path.exists(path):
        shutil.rmtree(path)

    worker = DEHBObjectiveFuncWrapper(
        obj_func=Wrapper(obj_func),
        n_workers=n_workers,
        continual_max_fidel=obj_func.max_fidel,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=100,
        save_dir_name=save_dir_name,
        obj_keys=["fitness"],
        runtime_key="cost",
        fidel_keys=["epoch"],
        max_total_eval_time=max_total_eval_time,
    )
    assert worker.fidel_keys == ["epoch"]
    assert worker.runtime_key == "cost"
    assert worker.n_actual_evals_in_opt == n_actual_evals_in_opt
    assert worker.n_workers == n_workers

    dehb = DEHB(
        f=worker,
        cs=obj_func.config_space,
        dimensions=len(obj_func.config_space),
        min_budget=obj_func.min_fidel,
        max_budget=obj_func.max_fidel,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path=log_file_name,
    )
    dehb.run(fevals=n_actual_evals_in_opt)
    out = worker.get_results()["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    if max_total_eval_time < 1e9:
        assert len(out) <= 70  # terminated before 100 evals
        assert np.all(np.asarray(out[:-n_workers]) <= 50)
    else:
        assert len(out) >= 100

    shutil.rmtree(path)
    shutil.rmtree(log_file_name)


@pytest.mark.parametrize("max_total_eval_time", (np.inf, 50))
def test_dehb(max_total_eval_time):
    n_workers = 4 if IS_LOCAL else 2  # github actions has only 2 cores
    run_dehb(n_workers, max_total_eval_time=max_total_eval_time)


if __name__ == "__main__":
    unittest.main()
