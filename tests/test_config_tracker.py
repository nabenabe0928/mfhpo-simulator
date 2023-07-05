from __future__ import annotations

import json
import pytest
import unittest

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper
from benchmark_simulator._trackers._config_tracker import _two_dicts_almost_equal

from tests.utils import dummy_func, cleanup, SUBDIR_NAME


class Optimizer(AbstractAskTellOptimizer):
    def __init__(self, valid: bool = True):
        self._count = 0
        if valid:
            self._configs = [0, 0, 0, 0, 1, 1]
            self._fidels = [1, 3, 9, 10, 10, 10]
            self._config_ids = [0, 0, 0, 1, 2, 3]
            self._ans = [0, 1, 3, 0, 0, 0]
        else:
            self._configs = [0, 1]
            self._fidels = [10, 10]
            self._config_ids = [0, 0]

        self._n_evals = len(self._configs)

    def ask(self):
        config, fidels = {"x": self._configs[self._count]}, {"epoch": self._fidels[self._count]}
        config_id = self._config_ids[self._count]
        self._count += 1
        return config, fidels, config_id

    def tell(self, *args, **kwargs):
        pass

    def optimize(self, func):
        for _ in range(self._n_evals):
            config, fidels, config_id = self.ask()
            func(eval_config=config, fidels=fidels, config_id=config_id)


def test_two_dicts_almost_equal():
    d1, d2 = {"x": 1.0}, {"x": 1.0 + 1e-12}
    assert _two_dicts_almost_equal(d1, d2)
    assert d1 != d2
    d2["y"] = 1
    assert not _two_dicts_almost_equal(d1, d2)
    d1, d2 = {"x": 1.0}, {"x": 2.0}
    assert not _two_dicts_almost_equal(d1, d2)


def run_opt(wrapper, opt, ask_and_tell):
    if ask_and_tell:
        wrapper.simulate(opt)
    else:
        opt.optimize(func=wrapper)


@cleanup
def optimize(ask_and_tell: bool, valid: bool):
    opt = Optimizer(valid=valid)
    wrapper = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        save_dir_name=SUBDIR_NAME,
        n_workers=1,
        continual_max_fidel=10,
        fidel_keys=["epoch"],
        ask_and_tell=ask_and_tell,
        store_config=True,
        n_evals=opt._n_evals,
        n_actual_evals_in_opt=opt._n_evals + 1,
    )
    if not valid:
        with pytest.raises(ValueError, match=r".*got the duplicated config_id.*"):
            run_opt(wrapper, opt, ask_and_tell)

        return
    else:
        run_opt(wrapper, opt, ask_and_tell)

    with open(wrapper._main_wrapper._paths.result, mode="r") as f:
        out = json.load(f)["prev_fidel"]

    assert out == opt._ans


@pytest.mark.parametrize("ask_and_tell", (True, False))
@pytest.mark.parametrize("valid", (True, False))
def test_config_tracker(ask_and_tell: bool, valid: bool):
    optimize(ask_and_tell=ask_and_tell, valid=valid)


if __name__ == "__main__":
    unittest.main()
