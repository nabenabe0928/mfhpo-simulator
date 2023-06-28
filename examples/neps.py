from __future__ import annotations

import os
import warnings
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWrapper

import neps

import numpy as np

from examples.utils import get_bench_instance, get_subdir_name, parse_args


class NEPSWorker(ObjectiveFuncWrapper):
    def __call__(self, **eval_config: dict[str, Any]) -> dict[str, float]:
        _eval_config = eval_config.copy()
        fidel_key = self.fidel_keys[0]
        fidels = {fidel_key: _eval_config.pop(fidel_key)}
        return super().__call__(eval_config=_eval_config, fidels=fidels)


def get_pipeline_space(config_space: CS.ConfigurationSpace) -> dict[str, neps.search_spaces.parameter.Parameter]:
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
    fidel_key: str,
    n_workers: int = 4,
    n_actual_evals_in_opt: int = 455,
    seed: int = 42,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
):
    np.random.seed(seed)
    worker = NEPSWorker(
        subdir_name=subdir_name,
        launch_multiple_wrappers_from_user_side=True,
        n_workers=n_workers,
        obj_func=obj_func,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        fidel_keys=[fidel_key],
        continual_max_fidel=max_fidel,
        seed=seed,
    )
    pipeline_space = get_pipeline_space(config_space)
    pipeline_space[fidel_key] = neps.IntegerParameter(lower=min_fidel, upper=max_fidel, is_fidelity=True)

    neps.run(
        run_pipeline=worker,
        pipeline_space=pipeline_space,
        root_directory="neps-log",
        max_evaluations_total=n_actual_evals_in_opt,
    )


if __name__ == "__main__":
    if os.path.exists("neps-log"):
        warnings.warn(
            "If `neps-log` already exists, NePS continues the optimization using the log, \n"
            "so pleaase remove the `neps-log` if you would like to start the optimization from scratch."
        )

    args = parse_args()
    subdir_name = get_subdir_name(args)
    bench = get_bench_instance(args)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"

    run_neps(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        subdir_name=os.path.join("neps", subdir_name),
        seed=args.seed,
    )
