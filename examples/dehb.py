import os
from typing import Any, Dict, List

import ConfigSpace as CS

from benchmark_simulator import CentralWorkerManager

from dehb import DEHB

import numpy as np

from examples.utils import get_bench_instance, get_subdir_name, parse_args


class DEHBCentralWorkerManager(CentralWorkerManager):
    # Adapt to the DEHB interface at https://github.com/automl/DEHB/
    def __call__(self, config: CS.Configuration, budget: int, **data_to_scatter: Any) -> Dict[str, float]:
        eval_config = config.get_dictionary()
        results = super().__call__(eval_config=eval_config, fidel=budget, **data_to_scatter)
        return dict(fitness=results[self.obj_keys[0]], cost=results[self.runtime_key])


def run_dehb(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    subdir_name: str,
    min_fidel: int,
    max_fidel: int,
    n_workers: int = 4,
    n_actual_evals_in_opt: int = 455,
    obj_keys: List[str] = ["loss"][:],
    runtime_key: str = "runtime",
    seed: int = 42,
    continual_eval: bool = True,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    np.random.seed(seed)
    manager = DEHBCentralWorkerManager(
        subdir_name=subdir_name,
        n_workers=n_workers,
        obj_func=obj_func,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        max_fidel=max_fidel,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        seeds=[seed] * n_workers,
        continual_eval=continual_eval,
    )

    dehb = DEHB(
        f=manager,
        cs=config_space,
        dimensions=len(config_space),
        min_budget=min_fidel,
        max_budget=max_fidel,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/",
    )
    data_to_scatter = {}
    if hasattr(obj_func, "get_benchdata"):
        # This data is shared in memory, and thus the optimization becomes quicker!
        data_to_scatter = {"benchdata": obj_func.get_benchdata()}

    dehb.run(fevals=n_actual_evals_in_opt, **data_to_scatter)


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    bench = get_bench_instance(args, keep_benchdata=False)

    run_dehb(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidel,
        max_fidel=bench.max_fidel,
        n_workers=args.n_workers,
        subdir_name=os.path.join("dehb", subdir_name),
        seed=args.seed,
    )
