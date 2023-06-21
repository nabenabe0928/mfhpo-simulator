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
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        fidel_keys: list[str],
        min_fidels: dict[str, int | float],
        max_fidels: dict[str, int | float],
        discrete: bool = False,
    ):
        self._config_space = config_space
        self._fidel_keys = fidel_keys
        self._min_fidels = min_fidels
        self._max_fidels = max_fidels
        self._discrete = discrete

    def ask(self) -> dict[str, Any]:
        config = self._config_space.sample_configuration().get_dictionary()
        return {k: int(v * 10) / 10 for k, v in config.items()} if self._discrete else config


class RandomOptimizerWrapper(AbstractAskTellOptimizer):
    def __init__(self, opt: RandomOptimizer, very_random: bool = False):
        self._opt = opt
        self._fidel_keys = self._opt._fidel_keys
        self._min_fidels = self._opt._min_fidels
        self._max_fidels = self._opt._max_fidels
        self._rng = np.random.RandomState(0)
        self._very_random = very_random

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        eval_config = self._opt.ask()
        fidels = (
            {
                k: self._rng.randint(self._max_fidels[k] - self._min_fidels[k]) + self._min_fidels[k]
                for k in self._fidel_keys
            }
            if self._very_random
            else self._opt._max_fidels
        )
        return eval_config, fidels

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
        trial_id: int,
    ) -> None:
        pass


def test_random_with_ask_and_tell():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            fidel_keys=bench.fidel_keys,
            min_fidels=bench.min_fidels,
            max_fidels=bench.max_fidels,
        ),
    )
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=10,
        obj_func=bench,
        n_actual_evals_in_opt=411,
        n_evals=400,
        seed=0,
        fidel_keys=bench.fidel_keys,
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    shutil.rmtree(worker.dir_name)


def test_random_with_ask_and_tell_store_config():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            fidel_keys=bench.fidel_keys,
            min_fidels=bench.min_fidels,
            max_fidels=bench.max_fidels,
        ),
    )
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=10,
        obj_func=bench,
        n_actual_evals_in_opt=411,
        n_evals=400,
        seed=0,
        fidel_keys=bench.fidel_keys,
        store_config=True,
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    assert all(k in list(out.keys()) for k in bench.config_space)
    for k in out.keys():
        assert diffs.size == len(out[k])

    shutil.rmtree(worker.dir_name)


def test_random_with_ask_and_tell_continual_eval():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            fidel_keys=bench.fidel_keys,
            min_fidels=bench.min_fidels,
            max_fidels=bench.max_fidels,
            discrete=True,
        ),
        very_random=True,
    )
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=10,
        obj_func=bench,
        n_actual_evals_in_opt=411,
        n_evals=400,
        seed=0,
        fidel_keys=bench.fidel_keys,
        continual_max_fidel=bench.max_fidels["z0"],
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    for k in out.keys():
        assert diffs.size == len(out[k])

    shutil.rmtree(worker.dir_name)


def test_random_with_ask_and_tell_many_parallel():
    subdir_name = "test-mfbranin-ask-and-tell"
    bench = MFBranin()
    opt = RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            fidel_keys=bench.fidel_keys,
            min_fidels=bench.min_fidels,
            max_fidels=bench.max_fidels,
        ),
    )
    worker = AskTellWorkerManager(
        subdir_name=subdir_name,
        n_workers=1000,
        obj_func=bench,
        n_actual_evals_in_opt=11001,
        n_evals=10000,
        fidel_keys=bench.fidel_keys,
        seed=0,
    )
    worker.simulate(opt)
    out = json.load(open(os.path.join(worker.dir_name, "results.json")))
    diffs = out["cumtime"] - np.maximum.accumulate(out["cumtime"])
    assert np.allclose(diffs, 0.0)
    shutil.rmtree(worker.dir_name)


if __name__ == "__main__":
    unittest.main()
