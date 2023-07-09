from __future__ import annotations

import shutil
import unittest
from typing import Any

from benchmark_apis import MFBranin

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import ConfigSpace as CS

import numpy as np


DEFAULT_KWARGS = dict(
    save_dir_name="test-mfbranin-ask-and-tell",
    ask_and_tell=True,
    n_workers=10,
    n_actual_evals_in_opt=411,
    n_evals=400,
)


class RandomOptimizer:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        fidel_keys: list[str],
        min_fidels: dict[str, int | float],
        max_fidels: dict[str, int | float],
        discrete: bool,
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
    def __init__(self, opt: RandomOptimizer, very_random: bool):
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
        return eval_config, fidels, None

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
        config_id: int | None,
    ) -> None:
        pass


def fetch_randopt_wrapper(bench: MFBranin, discrete: bool = False, very_random: bool = False) -> RandomOptimizerWrapper:
    return RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            fidel_keys=bench.fidel_keys,
            min_fidels=bench.min_fidels,
            max_fidels=bench.max_fidels,
            discrete=discrete,
        ),
        very_random=very_random,
    )


def optimize(n_evals: int = 400, discrete: bool = False, very_random: bool = False, **obj_kwd):
    kwargs = DEFAULT_KWARGS.copy()
    if n_evals > 1000:
        kwargs.update(n_workers=1000, n_actual_evals_in_opt=11001, n_evals=10000)

    bench = MFBranin()
    opt = fetch_randopt_wrapper(bench=bench, discrete=discrete, very_random=very_random)
    worker = ObjectiveFuncWrapper(obj_func=bench, fidel_keys=bench.fidel_keys, **kwargs, **obj_kwd)
    worker.simulate(opt)
    out = worker.get_results()
    shutil.rmtree(worker.dir_name)
    if "max_total_eval_time" not in obj_kwd:
        assert len(out["cumtime"]) >= worker._main_wrapper._wrapper_vars.n_evals

    return out


def test_random_with_ask_and_tell():
    out = optimize()["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)


def test_random_with_ask_and_tell_with_max_total_eval_time():
    out = optimize(max_total_eval_time=3600 * 20)["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    assert len(out) < 300  # terminated by time limit


def test_random_with_ask_and_tell_store_config():
    out = optimize(store_config=True)
    diffs = np.abs(out["cumtime"] - np.maximum.accumulate(out["cumtime"]))
    assert np.allclose(diffs, 0.0)
    assert all(k in list(out.keys()) for k in MFBranin().config_space)
    for k in out.keys():
        assert diffs.size == len(out[k])


def test_random_with_ask_and_tell_continual_eval():
    out = optimize(discrete=True, very_random=True, continual_max_fidel=MFBranin().max_fidels["z0"])
    diffs = np.abs(out["cumtime"] - np.maximum.accumulate(out["cumtime"]))
    assert np.allclose(diffs, 0.0)
    for k in out.keys():
        assert diffs.size == len(out[k])


def test_random_with_ask_and_tell_many_parallel():
    out = optimize(n_evals=10000)["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)


if __name__ == "__main__":
    unittest.main()
