from __future__ import annotations

import os
import warnings
from argparse import ArgumentParser
from typing import Any

import ConfigSpace as CS

from benchmark_apis.synthetic.branin import MFBranin
from benchmark_simulator import ObjectiveFuncWrapper

import neps


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


def parse_n_workers_and_worker_index() -> tuple[int, int]:
    parser = ArgumentParser()
    # As each process is separately excuted via commandline, we must receive the n_workers from commandline.
    parser.add_argument("--n_workers", type=int, required=4)
    parser.add_argument("--worker_index", type=int, default=None)
    args = parser.parse_args()
    return args.n_workers, args.worker_index


if __name__ == "__main__":
    log_file_name = "neps-log"
    if os.path.exists(log_file_name):
        warnings.warn(
            f"If `{log_file_name}` already exists, NePS continues the optimization using the log, \n"
            f"so pleaase remove the `{log_file_name}` if you would like to start the optimization from scratch."
        )

    bench, fidel_key = MFBranin(), "z0"
    n_workers, worker_index = parse_n_workers_and_worker_index()

    # 1. Define a wrapper instance
    worker = NEPSWorker(
        obj_func=bench,
        save_dir_name="neps-minimal",  # subdir is required because we launch multiple workers in different processes
        launch_multiple_wrappers_from_user_side=True,
        fidel_keys=[fidel_key],
        n_workers=n_workers,
        worker_index=worker_index,
    )
    pipeline_space = get_pipeline_space(bench.config_space)
    pipeline_space[fidel_key] = neps.IntegerParameter(
        lower=bench.min_fidels[fidel_key], upper=bench.max_fidels[fidel_key], is_fidelity=True
    )

    neps.run(
        # 2. Feed the wrapped objective function to the optimizer directly
        # 3. just start the optimization!
        run_pipeline=worker,
        pipeline_space=pipeline_space,
        root_directory=log_file_name,
        max_evaluations_total=worker.n_actual_evals_in_opt,
    )
