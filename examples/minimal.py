from __future__ import annotations

import ConfigSpace as CS

from benchmark_apis.synthetic.branin import MFBranin
from benchmark_simulator import ObjectiveFuncWrapper

from dehb import DEHB


class DEHBObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # 0. Adapt the manager.__call__ to the DEHB interface at https://github.com/automl/DEHB/
    def __call__(self, config: CS.Configuration, budget: int) -> dict[str, float]:
        eval_config = config.get_dictionary()
        results = super().__call__(eval_config=eval_config, fidels={self.fidel_keys[0]: int(budget)})
        return dict(fitness=results[self.obj_keys[0]], cost=results[self.runtime_key])


if __name__ == "__main__":
    bench, fidel_key = MFBranin(), "z0"
    config_space, min_fidel, max_fidel = bench.config_space, bench.min_fidels[fidel_key], bench.max_fidels[fidel_key]

    # 1. Define the manager instance
    manager = DEHBObjectiveFuncWrapper(obj_func=bench, continual_max_fidel=max_fidel, fidel_keys=[fidel_key])

    DEHB(
        # 2. Feed the wrapped objective function to the optimizer directly
        f=manager,
        cs=config_space,
        min_budget=min_fidel,
        max_budget=max_fidel,
        n_workers=manager.n_workers,
        output_path="dehb-log/",
    ).run(
        # 3. just start the optimization!
        fevals=manager.n_actual_evals_in_opt
    )
