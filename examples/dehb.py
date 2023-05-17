import os
import shutil
from typing import Any, Callable, Dict, Optional

import ConfigSpace as CS

import numpy as np

from dehb import DEHB

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import CentralWorkerManager
from examples.toy import TestFunc


class Wrapper:
    def __init__(self, bench: Any):
        self._bench = bench

    def __call__(
        self, eval_config: Dict[str, Any], fidel: int, seed: Optional[int], **data_to_scatter: Any
    ) -> Dict[str, Any]:
        output = self._bench(eval_config, fidel, seed, **data_to_scatter)
        ret_vals = dict(fitness=output["loss"], cost=output["runtime"])
        return ret_vals


class DEHBCentralWorkerManager(CentralWorkerManager):
    def __call__(self, config: Dict[str, Any], budget: int, **data_to_scatter: Any) -> Dict[str, float]:
        return super().__call__(eval_config=config, fidel=budget)


def run_dehb(
    obj_func: Callable,
    config_space: CS.ConfigurationSpace,
    min_fidel: int,
    max_fidel: int,
    n_workers: int,
    subdir_name: str,
    max_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    n_actual_evals_in_opt = max_evals + n_workers
    worker = DEHBCentralWorkerManager(
        obj_func=obj_func,
        n_workers=n_workers,
        max_fidel=max_fidel,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=max_evals,
        subdir_name=subdir_name,
        obj_keys=["fitness"],
        runtime_key="cost",
    )

    dehb = DEHB(
        f=worker,
        cs=config_space,
        dimensions=len(config_space),
        min_budget=min_fidel,
        max_budget=max_fidel,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/",
    )
    # kwargs = obj_func.get_shared_data()
    kwargs = {}
    dehb.run(fevals=n_actual_evals_in_opt, **kwargs)


if __name__ == "__main__":
    np.random.RandomState(0)
    bench = TestFunc()
    wrapped_func = Wrapper(bench)

    subdir_name = "example_dehb"
    dir_name = os.path.join(DIR_NAME, subdir_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    run_dehb(
        obj_func=wrapped_func,
        config_space=bench.config_space,
        min_fidel=bench.min_fidel,
        max_fidel=bench.max_fidel,
        n_workers=4,
        subdir_name=subdir_name,
    )
