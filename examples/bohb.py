from __future__ import annotations

import os
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWrapper, get_multiple_wrappers

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

import numpy as np

from examples.utils import get_bench_instance, get_save_dir_name, parse_args


class BOHBWorker(Worker):
    # https://github.com/automl/HpBandSter
    def __init__(self, worker: ObjectiveFuncWrapper, sleep_interval: int = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker

    def compute(self, config: dict[str, Any], budget: int, **kwargs: Any) -> dict[str, float]:
        fidel_keys = self._worker.fidel_keys
        fidels = dict(epoch=int(budget)) if "epoch" in fidel_keys else {k: int(budget) for k in fidel_keys}
        # config_id: a triplet of ints(iteration, budget index, running index) internally used in BOHB
        # By passing config_id, it increases the safety in the continual learning
        config_id = kwargs["config_id"][0] + 100000 * kwargs["config_id"][2]
        results = self._worker(eval_config=config, fidels=fidels, config_id=config_id)
        return dict(loss=results["loss"])


def get_bohb_workers(
    run_id: str,
    ns_host: str,
    obj_func: Any,
    save_dir_name: str,
    max_fidel: int,
    fidel_key: str,
    n_workers: int,
    n_actual_evals_in_opt: int,
    n_evals: int,
    seed: int,
) -> list[BOHBWorker]:
    kwargs = dict(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        continual_max_fidel=max_fidel,
        fidel_keys=[fidel_key],
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
    )
    bohb_workers = []
    for i, w in enumerate(get_multiple_wrappers(**kwargs)):
        worker = BOHBWorker(worker=w, id=i, nameserver=ns_host, run_id=run_id)
        worker.run(background=True)
        bohb_workers.append(worker)

    return bohb_workers


def run_bohb(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: str,
    n_workers: int = 4,
    n_actual_evals_in_opt: int = 455,
    seed: int = 42,
    run_id: str = "bohb-run",
    ns_host: str = "127.0.0.1",
    n_evals: int = 450,  # eta=3,S=2,100 full evals
    n_brackets: int = 70,  # 22 HB iter --> 33 SH brackets
) -> None:
    ns = hpns.NameServer(run_id=run_id, host=ns_host, port=None)
    ns.start()
    _ = get_bohb_workers(
        run_id=run_id,
        ns_host=ns_host,
        obj_func=obj_func,
        save_dir_name=save_dir_name,
        max_fidel=max_fidel,
        fidel_key=fidel_key,
        n_workers=n_workers,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
    )
    bohb = BOHB(
        configspace=config_space,
        run_id=run_id,
        min_budget=min_fidel,
        max_budget=max_fidel,
    )
    bohb.run(n_iterations=n_brackets, min_n_workers=n_workers)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    np.random.seed(args.seed)
    obj_func = get_bench_instance(args)

    run_id = f"bench={args.bench_name}_dataset={args.dataset_id}_nworkers={args.n_workers}_seed={args.seed}"
    fidel_key = "epoch" if "epoch" in obj_func.fidel_keys else "z0"
    run_bohb(
        obj_func=obj_func,
        config_space=obj_func.config_space,
        min_fidel=obj_func.min_fidels[fidel_key],
        max_fidel=obj_func.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        save_dir_name=os.path.join("bohb", save_dir_name),
    )
