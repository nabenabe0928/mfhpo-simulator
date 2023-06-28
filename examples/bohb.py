from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWrapper

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

import numpy as np

from examples.utils import get_bench_instance, get_subdir_name, parse_args


class BOHBWorker(Worker):
    # https://github.com/automl/HpBandSter
    def __init__(self, worker: ObjectiveFuncWrapper, sleep_interval: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker

    def compute(self, config: dict[str, Any], budget: int, **kwargs: Any) -> dict[str, float]:
        fidel_keys = self._worker.fidel_keys
        fidels = dict(epoch=int(budget)) if "epoch" in fidel_keys else {k: int(budget) for k in fidel_keys}
        results = self._worker(eval_config=config, fidels=fidels)
        return dict(loss=results["loss"])


def get_bohb_workers(
    run_id: str,
    ns_host: str,
    obj_func: Any,
    subdir_name: str,
    max_fidel: int,
    fidel_key: str,
    n_workers: int,
    n_actual_evals_in_opt: int,
    n_evals: int,
    seed: int,
) -> list[BOHBWorker]:
    kwargs = dict(
        obj_func=obj_func,
        launch_multiple_wrappers_from_user_side=True,
        n_workers=n_workers,
        subdir_name=subdir_name,
        continual_max_fidel=max_fidel,
        fidel_keys=[fidel_key],
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
    )

    pool = Pool()
    results = []
    for _ in range(n_workers):
        results.append(pool.apply_async(ObjectiveFuncWrapper, kwds=kwargs))

    pool.close()
    pool.join()

    workers = [result.get() for result in results]
    bohb_workers = []
    kwargs = dict(sleep_interval=0.5, nameserver=ns_host, run_id=run_id)
    for i in range(n_workers):
        worker = BOHBWorker(worker=workers[i], id=i, **kwargs)
        worker.run(background=True)
        bohb_workers.append(worker)

    return bohb_workers


def run_bohb(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    subdir_name: str,
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
        subdir_name=subdir_name,
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
    subdir_name = get_subdir_name(args)
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
        subdir_name=os.path.join("bohb", subdir_name),
    )
