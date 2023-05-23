import os
from typing import Any, Dict, List, Optional

import ConfigSpace as CS

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

from benchmark_simulator import CentralWorkerManager

from examples.utils import get_bench_instance, get_subdir_name, parse_args


class SMACCentralWorkerManager(CentralWorkerManager):
    def __call__(
        self,
        eval_config: Dict[str, Any],
        budget: int,
        seed: Optional[int] = None,
        data_to_scatter: Optional[Dict[str, Any]] = None,
    ) -> float:
        data_to_scatter = {} if data_to_scatter is None else data_to_scatter
        output = super().__call__(eval_config, {self.fidel_keys[0]: int(budget)}, **data_to_scatter)
        return output[self._obj_keys[0]]


def run_smac(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    subdir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: List[str],
    n_workers: int = 4,
    n_init: int = 5,
    n_actual_evals_in_opt: int = 455,
    obj_keys: List[str] = ["loss"][:],
    runtime_key: str = "runtime",
    seed: int = 42,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    scenario = Scenario(
        config_space,
        n_trials=n_actual_evals_in_opt,
        min_budget=min_fidel,
        max_budget=max_fidel,
        n_workers=n_workers,
    )
    worker = SMACCentralWorkerManager(
        obj_func=obj_func,
        n_workers=n_workers,
        subdir_name=subdir_name,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        seed=seed,
        fidel_keys=[fidel_key],
        continual_max_fidel=max_fidel,
    )
    smac = MFFacade(
        scenario,
        worker,
        initial_design=MFFacade.get_initial_design(scenario, n_configs=n_init),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    data_to_scatter = {}
    if hasattr(obj_func, "get_benchdata"):
        # This data is shared in memory, and thus the optimization becomes quicker!
        data_to_scatter = {"benchdata": obj_func.get_benchdata()}

    # data_to_scatter must be a keyword argument.
    smac.optimize(data_to_scatter=data_to_scatter)


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    bench = get_bench_instance(args, keep_benchdata=False)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"
    run_smac(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        subdir_name=os.path.join("smac", subdir_name),
        seed=args.seed,
    )
