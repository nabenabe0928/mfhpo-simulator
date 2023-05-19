import os
from multiprocessing import Pool
from typing import Any, Dict, List

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWorker

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

import numpy as np

from optimizers.utils import get_bench_instance, get_subdir_name, parse_args


class BOHBWorker(Worker):
    # https://github.com/automl/HpBandSter
    def __init__(self, worker: ObjectiveFuncWorker, sleep_interval: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker

    def compute(self, config: Dict[str, Any], budget: int, **kwargs: Any) -> Dict[str, float]:
        results = self._worker(eval_config=config, fidel=budget)
        return dict(loss=results["loss"])


def get_bohb_workers(
    run_id: str,
    ns_host: str,
    obj_func: Any,
    subdir_name: str,
    max_fidel: int,
    n_workers: int,
    n_actual_evals_in_opt: int,
    n_evals: int,
    obj_keys: List[str],
    runtime_key: str,
    seed: int,
    continual_eval: bool,
) -> List[BOHBWorker]:
    kwargs = dict(
        obj_func=obj_func,
        n_workers=n_workers,
        subdir_name=subdir_name,
        max_fidel=max_fidel,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
        continual_eval=continual_eval,
    )

    pool = Pool()
    results = []
    for _ in range(n_workers):
        results.append(pool.apply_async(ObjectiveFuncWorker, kwds=kwargs))

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
    n_workers: int = 4,
    n_actual_evals_in_opt: int = 455,
    obj_keys: List[str] = ["loss"][:],
    runtime_key: str = "runtime",
    seed: int = 42,
    continual_eval: bool = True,
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
        n_workers=n_workers,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        seed=seed,
        continual_eval=continual_eval,
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
    run_bohb(
        obj_func=obj_func,
        config_space=obj_func.config_space,
        min_fidel=obj_func.min_fidel,
        max_fidel=obj_func.max_fidel,
        n_workers=args.n_workers,
        subdir_name=os.path.join("bohb", subdir_name),
    )
