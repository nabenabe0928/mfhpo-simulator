from __future__ import annotations

from typing import Any

import ConfigSpace as CS

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

from benchmark_apis.synthetic.branin import MFBranin
from benchmark_simulator import ObjectiveFuncWrapper


class SMACObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # 0. Adapt the callable of the objective function to the SMAC interface at https://github.com/automl/SMAC3/
    def __call__(
        self,
        config: CS.Configuration,
        budget: int,
        seed: int | None = None,
        data_to_scatter: dict[str, Any] | None = None,
    ) -> float:
        eval_config = config.get_dictionary()
        output = super().__call__(eval_config, fidels={self.fidel_keys[0]: int(budget)})
        return output[self.obj_keys[0]]


if __name__ == "__main__":
    bench, fidel_key = MFBranin(), "z0"

    # 1. Define a wrapper instance (Default is n_workers=4, but you can change it from the argument)
    wrapper = SMACObjectiveFuncWrapper(obj_func=bench, fidel_keys=[fidel_key])
    scenario = Scenario(
        bench.config_space,
        n_trials=wrapper.n_actual_evals_in_opt,
        min_budget=bench.min_fidels[fidel_key],
        max_budget=bench.max_fidels[fidel_key],
        n_workers=wrapper.n_workers,
    )

    smac = MFFacade(
        scenario,
        # 2. Feed the wrapped objective function to the optimizer directly
        wrapper,
        initial_design=MFFacade.get_initial_design(scenario, n_configs=5),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    smac.optimize()
