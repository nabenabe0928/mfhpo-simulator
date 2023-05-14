from typing import Dict, Optional

import ConfigSpace as CS


class TestFunc:
    def __call__(self, eval_config: Dict[str, float], fidel: int, seed: Optional[int] = None):
        return dict(loss=eval_config["x"] ** 2, runtime=fidel / 1.0)

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
