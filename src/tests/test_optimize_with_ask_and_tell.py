from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
import unittest

from benchmark_apis import MFBranin
import numpy as np

from src import AbstractAskTellOptimizer
from src import ObjectiveFuncWrapper


if TYPE_CHECKING:
    import ConfigSpace as CS


DEFAULT_KWARGS = dict(
    n_workers=10,
    n_evals=400,
)


class RandomOptimizer:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        discrete: bool,
    ):
        self._config_space = config_space
        self._discrete = discrete

    def ask(self) -> dict[str, Any]:
        config = self._config_space.sample_configuration().get_dictionary()
        return {k: int(v * 10) / 10 for k, v in config.items()} if self._discrete else config


class RandomOptimizerWrapper(AbstractAskTellOptimizer):
    def __init__(self, opt: RandomOptimizer):
        self._opt = opt

    def ask(self) -> dict[str, Any]:
        return self._opt.ask()

    def tell(
        self,
        eval_config: dict[str, Any],
        results: list[float],
    ) -> None:
        pass


def _wrap_bench(bench: MFBranin):
    max_fidels = bench.max_fidels

    def wrapped(eval_config: dict[str, Any], **kwargs: Any) -> list[float]:
        out = bench(eval_config, fidels=max_fidels)
        return [out["loss"], out["runtime"]]

    return wrapped


def fetch_randopt_wrapper(bench: MFBranin, discrete: bool = False) -> RandomOptimizerWrapper:
    return RandomOptimizerWrapper(
        RandomOptimizer(
            config_space=bench.config_space,
            discrete=discrete,
        ),
    )


def optimize(n_evals: int = 400, discrete: bool = False, **obj_kwd):
    kwargs = DEFAULT_KWARGS.copy()
    if n_evals > 1000:
        kwargs.update(n_workers=1000, n_evals=10000)

    bench = MFBranin()
    opt = fetch_randopt_wrapper(bench=bench, discrete=discrete)
    worker = ObjectiveFuncWrapper(obj_func=_wrap_bench(bench), **kwargs, **obj_kwd)
    worker.simulate(opt)
    out = worker.get_results()
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


def test_random_with_ask_and_tell_many_parallel():
    out = optimize(n_evals=10000)["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)


if __name__ == "__main__":
    unittest.main()
