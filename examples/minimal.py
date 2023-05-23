from typing import Dict

import ConfigSpace as CS

from benchmark_apis.synthetic.branin import MFBranin
from benchmark_simulator import CentralWorkerManager

from dehb import DEHB


class DEHBCentralWorkerManager(CentralWorkerManager):
    # 0. Adapt the manager.__call__ to the DEHB interface at https://github.com/automl/DEHB/
    def __call__(self, config: CS.Configuration, budget: int) -> Dict[str, float]:
        eval_config = config.get_dictionary()
        results = super().__call__(eval_config=eval_config, fidels={self.fidel_keys[0]: int(budget)})
        return dict(fitness=results[self.obj_keys[0]], cost=results[self.runtime_key])


if __name__ == "__main__":
    bench, n_workers, n_actual_evals_in_opt, fidel_key = MFBranin(), 4, 105, "z0"
    config_space, min_fidel, max_fidel = bench.config_space, bench.min_fidels[fidel_key], bench.max_fidels[fidel_key]

    # 1. Define the manager instance
    manager = DEHBCentralWorkerManager(
        subdir_name="dehb-minimal",
        n_workers=n_workers,
        obj_func=bench,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_actual_evals_in_opt - n_workers - 1,
        continual_max_fidel=max_fidel,
        fidel_keys=[fidel_key],
    )

    optimizer_kwargs = dict(
        cs=config_space, min_budget=min_fidel, max_budget=max_fidel, n_workers=n_workers, output_path="dehb-log/"
    )
    DEHB(
        # 2. Feed the wrapped objective function to the optimizer directly
        f=manager,
        **optimizer_kwargs,
    ).run(
        # 3. just start the optimization!
        fevals=n_actual_evals_in_opt
    )
