from __future__ import annotations

import json
import os
import shutil
import unittest
from typing import Any

from benchmark_apis import MFBranin

from benchmark_simulator import AbstractAskTellOptimizer, AskTellWorkerManager

import ConfigSpace as CS

import numpy as np


class RandomOptimizer:
    def __init__(self, config_space: CS.ConfigurationSpace, max_fidels: dict[str, int | float]):
        self._config_space = config_space
        self._max_fidels = max_fidels

    def ask(self) -> dict[str, Any]:
        return self._config_space.sample_configuration().get_dictionary()


class RandomOptimizerWrapper(AbstractAskTellOptimizer):
    def __init__(self, opt: RandomOptimizer):
        self._opt = opt

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        eval_config = self._opt.ask()
        return eval_config, self._opt._max_fidels

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
    ) -> None:
        pass


def test_random_with_ask_and_tell():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(RandomOptimizer(bench.config_space, bench.max_fidels))
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=10,
        obj_func=bench,
        n_actual_evals_in_opt=411,
        n_evals=400,
        seed=0,
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    shutil.rmtree(worker.dir_name)


def test_random_with_ask_and_tell_many_parallel():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(RandomOptimizer(bench.config_space, bench.max_fidels))
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=1000,
        obj_func=bench,
        n_actual_evals_in_opt=11001,
        n_evals=10000,
        seed=0,
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    shutil.rmtree(worker.dir_name)


if __name__ == "__main__":
    unittest.main()
