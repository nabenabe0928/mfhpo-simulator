import os
from typing import Any, Dict, List

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWorker

import neps

import numpy as np

from optimizers.utils import BENCH_CHOICES, get_subdir_name, parse_args


class NEPSWorker(ObjectiveFuncWorker):
    def __call__(self, **eval_config: Dict[str, Any]) -> Dict[str, float]:
        _eval_config = eval_config.copy()
        fidel = _eval_config.pop("fidel")  # Fidelity param name must be "fidel"
        return super().__call__(eval_config=_eval_config, fidel=fidel)


def get_pipeline_space(config_space: CS.ConfigurationSpace) -> Dict[str, neps.search_spaces.parameter.Parameter]:
    pipeline_space = {}
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter):
            pipeline_space[hp.name] = neps.FloatParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            pipeline_space[hp.name] = neps.IntegerParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            pipeline_space[hp.name] = neps.CategoricalParameter(choices=hp.choices)
        else:
            raise ValueError(f"{type(hp)} is not supported")

    return pipeline_space


def run_neps(
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
):
    np.random.seed(seed)
    worker = NEPSWorker(
        subdir_name=subdir_name,
        n_workers=n_workers,
        obj_func=obj_func,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        max_fidel=max_fidel,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        seed=seed,
        continual_eval=continual_eval,
    )
    pipeline_space = get_pipeline_space(config_space)
    # Fidelity param name must be "fidel"
    pipeline_space["fidel"] = neps.IntegerParameter(lower=min_fidel, upper=max_fidel, is_fidelity=True)

    neps.run(
        run_pipeline=worker,
        pipeline_space=pipeline_space,
        root_directory="neps-log",
        max_evaluations_total=n_actual_evals_in_opt,
    )


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    bench = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed)
    run_neps(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidel,
        max_fidel=bench.max_fidel,
        n_workers=args.n_workers,
        subdir_name=os.path.join("neps", subdir_name),
        seed=args.seed,
    )
