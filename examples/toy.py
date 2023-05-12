from typing import Dict

import ConfigSpace as CS


class TestFunc:
    def __call__(self, eval_config: Dict[str, float], budget: int, seed):
        return dict(loss=eval_config["x"]**2, runtime=budget / 1.0)

    @property
    def config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5.0, 5.0))
        return cs

    @property
    def min_budget(self):
        return 1

    @property
    def max_budget(self):
        return 9
