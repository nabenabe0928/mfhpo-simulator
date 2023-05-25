from __future__ import annotations

import os
import shutil
import unittest
from typing import Any

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import CentralWorkerManager

import ConfigSpace as CS

from dehb import DEHB

import numpy as np

import ujson as json


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


class DEHBCentralWorkerManager(CentralWorkerManager):
    def __call__(self, config: dict[str, Any], budget: int, **data_to_scatter: Any) -> dict[str, float]:
        return super().__call__(eval_config=config, fidels={"epoch": int(budget)})


def test_dehb():
    n_workers = 4 if os.system("hostname") == "EB-B9400CBA" else 2  # github actions has only 2 cores
    n_actual_evals_in_opt = 100 + n_workers
    subdir_name = "dummy"
    path = os.path.join(DIR_NAME, subdir_name)
    obj_func = ToyFunc()

    if os.path.exists(path):
        shutil.rmtree(path)

    worker = DEHBCentralWorkerManager(
        obj_func=Wrapper(obj_func),
        n_workers=n_workers,
        continual_max_fidel=obj_func.max_fidel,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=100,
        subdir_name=subdir_name,
        obj_keys=["fitness"],
        runtime_key="cost",
        fidel_keys=["epoch"],
    )

    dehb = DEHB(
        f=worker,
        cs=obj_func.config_space,
        dimensions=len(obj_func.config_space),
        min_budget=obj_func.min_fidel,
        max_budget=obj_func.max_fidel,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/",
    )
    dehb.run(fevals=n_actual_evals_in_opt)
    out = json.load(open(os.path.join(path, "results.json")))
    shutil.rmtree(path)
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    shutil.rmtree("dehb-log")


if __name__ == "__main__":
    unittest.main()
