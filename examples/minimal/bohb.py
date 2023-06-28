from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from benchmark_apis.synthetic.branin import MFBranin
from benchmark_simulator import ObjectiveFuncType, ObjectiveFuncWrapper, get_multiple_wrappers

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB


bohb_run_id = "bohb-run"
bohb_nameserver = "127.0.0.1"


class BOHBWorker(Worker):
    # 0. Adapt the objective function (`compute`) to the BOHB interface at https://github.com/automl/HpBandSter/
    def __init__(self, worker: ObjectiveFuncWrapper, sleep_interval: int = 0.5, **kwargs):
        super().__init__(nameserver=bohb_nameserver, run_id=bohb_run_id, **kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker

    def compute(self, config: dict[str, Any], budget: int, **kwargs: Any) -> dict[str, float]:
        fidel_keys = self._worker.fidel_keys
        fidels = dict(epoch=int(budget)) if "epoch" in fidel_keys else {k: int(budget) for k in fidel_keys}
        results = self._worker(eval_config=config, fidels=fidels)
        return dict(loss=results["loss"])


@contextmanager
def get_bohb(obj_func: ObjectiveFuncType, n_workers: int, fidel_key: str) -> BOHB:
    ns = hpns.NameServer(run_id=bohb_run_id, host=bohb_nameserver, port=None)
    ns.start()

    # 1. Define wrapper instances and run them on background
    bohb_workers = []
    for i, wrapper in enumerate(get_multiple_wrappers(n_workers=n_workers, obj_func=obj_func, fidel_keys=[fidel_key])):
        worker = BOHBWorker(worker=wrapper, id=i)
        worker.run(background=True)
        bohb_workers.append(worker)

    # 2. Instantiate an optimizer (BOHB is a bit tricky and it recognizes workers on background)
    bohb = BOHB(
        configspace=obj_func.config_space,
        run_id=bohb_run_id,
        min_budget=obj_func.min_fidels[fidel_key],
        max_budget=obj_func.max_fidels[fidel_key],
    )
    yield bohb

    bohb.shutdown()
    ns.shutdown()


if __name__ == "__main__":
    n_workers = 4
    with get_bohb(obj_func=MFBranin(), n_workers=n_workers, fidel_key="z0") as bohb:
        # 3. just start the optimization!
        bohb.run(n_iterations=20, min_n_workers=n_workers)
