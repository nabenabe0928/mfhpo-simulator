from __future__ import annotations

import time
from typing import Any
import unittest

import numpy as np
import pytest

from benchmark_simulator import AbstractAskTellOptimizer
from benchmark_simulator import ObjectiveFuncWrapper
from tests.utils import cleanup
from tests.utils import get_configs
from tests.utils import ON_UBUNTU
from tests.utils import simplest_dummy_func
from tests.utils import SUBDIR_NAME


class DummyOptimizer(AbstractAskTellOptimizer):
    def __init__(
        self,
        configs: np.ndarray,
        obj_func: ObjectiveFuncWrapper,
        n_workers: int,
        unittime: float,
    ):
        self._n_workers = n_workers
        self._n_evals = configs.size + self._n_workers
        self._obj_func = obj_func
        self._observations: list[dict[str, float]] = []
        self._unittime = unittime
        self._configs = configs[::-1].tolist()

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None, int | None]:
        waiting_time = (len(self._observations) + 1) * self._unittime
        if len(self._configs):
            time.sleep(waiting_time)
            return {"x": self._configs.pop()}, None, None
        else:
            return {"x": 10**5}, None, None

    def tell(self, eval_config: dict[str, Any], results: dict[str, float], *args, **kwargs) -> None:
        eval_config["loss"] = results["loss"]
        self._observations.append(eval_config.copy())


@cleanup
def optimize(index: int, n_workers: int):
    unittime = 1e-1 if ON_UBUNTU else 1.0
    configs, ans = get_configs(index=index, unittime=unittime)
    n_evals = configs.size
    wrapper = ObjectiveFuncWrapper(
        obj_func=simplest_dummy_func,
        n_workers=n_workers,
        ask_and_tell=True,
        n_actual_evals_in_opt=configs.size + n_workers,
        n_evals=configs.size,
        expensive_sampler=True,
        save_dir_name=SUBDIR_NAME,
    )
    opt = DummyOptimizer(
        configs=configs,
        n_workers=wrapper.n_workers,
        obj_func=wrapper,
        unittime=unittime,
    )
    wrapper.simulate(opt)
    diff = np.abs(np.array(wrapper.get_results()["cumtime"])[:n_evals] - ans)
    assert np.all(diff < unittime * 1.5)


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 6, 7, 8))
def test_opt(index: int) -> None:
    if index == 1:
        optimize(index=index, n_workers=2)
    elif ON_UBUNTU:
        optimize(index=index, n_workers=4)


if __name__ == "__main__":
    unittest.main()
