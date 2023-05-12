import os
import shutil
from typing import Any, Callable, Dict, Optional

import ConfigSpace as CS

import numpy as np

from dehb import DEHB

from benchmark_simulator._constants import DIR_NAME
from benchmark_simulator.simulator import CentralWorker
from examples.toy import TestFunc


def run_dehb(
    obj_func: Callable,
    config_space: CS.ConfigurationSpace,
    min_budget: int,
    max_budget: int,
    n_workers: int,
    subdir_name: str,
    max_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    worker = CentralWorker(
        obj_func=obj_func,
        n_workers=n_workers,
        max_budget=max_budget,
        max_evals=max_evals,
        subdir_name=subdir_name,
        loss_key="fitness",
        runtime_key="cost",
    )

    dehb = DEHB(
        f=worker,
        cs=config_space,
        dimensions=len(config_space),
        min_budget=min_budget,
        max_budget=max_budget,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/"
    )
    # kwargs = obj_func.get_shared_data()
    kwargs = {}
    dehb.run(fevals=max_evals, **kwargs)


class Wrapper:
    def __init__(self, bench: Any):
        self._bench = bench

    def __call__(
        self, eval_config: Dict[str, Any], budget: int, seed: Optional[int], **data_to_scatter: Any
    ) -> Dict[str, Any]:
        output = self._bench(eval_config, budget, seed, **data_to_scatter)
        ret_vals = dict(fitness=output["loss"], cost=output["runtime"])
        return ret_vals


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
        min_budget=bench.min_budget,
        max_budget=bench.max_budget,
        n_workers=4,
        subdir_name=subdir_name,
    )
