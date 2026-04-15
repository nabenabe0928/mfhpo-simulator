from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src._ask_tell_manager import _AskTellWorkerManager
from src._constants import _WrapperVars
from src._constants import AbstractAskTellOptimizer


if TYPE_CHECKING:
    from src._constants import ObjectiveFuncType


class ObjectiveFuncWrapper:
    """Objective function wrapper API for ask-and-tell optimizers.

    This class wraps an objective function and simulates multi-worker parallel optimization
    using an ask-and-tell interface. Each objective function call is not run in parallel;
    instead, cumulative runtimes are simulated internally per worker.

    Attributes:
        obj_keys (list[str]):
            The objective names that will be collected in results.
        runtime_key (str):
            The runtime name used to define the runtime of the user objective function.
        n_actual_evals_in_opt (int):
            The number of configuration evaluations during the actual optimization.
        n_workers (int):
            The number of (simulated) workers used in the optimization.

    Methods:
        simulate(opt: AbstractAskTellOptimizer) -> None:
            Run the optimization loop with the given ask-and-tell optimizer.
    """

    def __init__(
        self,
        obj_func: ObjectiveFuncType,
        n_workers: int = 4,
        n_actual_evals_in_opt: int = 105,
        n_evals: int = 100,
        obj_keys: list[str] | None = None,
        runtime_key: str = "runtime",
        seed: int | None = None,
        store_actual_cumtime: bool = False,
        allow_parallel_sampling: bool = False,
        config_tracking: bool = True,
        max_total_eval_time: float = np.inf,
        expensive_sampler: bool = False,
    ):
        """The initialization of a wrapper class for ask-and-tell optimization.

        Args:
            obj_func (ObjectiveFuncType):
                A callable object that serves as the objective function.
                Args:
                    eval_config: dict[str, Any]
                    seed: int | None
                    **data_to_scatter: Any
                Returns:
                    results: dict[str, float]
                        It must return `objective metric` and `runtime` at least.
            n_workers (int):
                The number of simulated workers. In other words, how many parallel workers to simulate.
            n_actual_evals_in_opt (int):
                The number of evaluations that optimizers do. Must be >= n_evals + n_workers.
            n_evals (int):
                How many configurations we would like to collect.
            obj_keys (list[str] | None):
                The keys of the objective metrics used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            seed (int | None):
                The random seed to be used to allocate random seed to each call.
            store_actual_cumtime (bool):
                Whether to store actual cumulative time at each point.
            allow_parallel_sampling (bool):
                Whether sampling can happen in parallel.
            config_tracking (bool):
                Whether to validate config_id provided from the user side.
            max_total_eval_time (float):
                The maximum total evaluation time for the optimization.
            expensive_sampler (bool):
                Whether the optimizer is expensive relative to a function evaluation.
        """
        self._n_workers = n_workers
        wrapper_vars = _WrapperVars(
            obj_func=obj_func,
            n_workers=n_workers,
            n_actual_evals_in_opt=n_actual_evals_in_opt,
            n_evals=n_evals,
            obj_keys=obj_keys if obj_keys is not None else ["loss"],
            runtime_key=runtime_key,
            seed=seed,
            max_total_eval_time=max_total_eval_time,
            store_actual_cumtime=store_actual_cumtime,
            allow_parallel_sampling=allow_parallel_sampling,
            config_tracking=config_tracking,
            expensive_sampler=expensive_sampler,
        )

        self._main_wrapper = _AskTellWorkerManager(wrapper_vars)

    @property
    def obj_keys(self) -> list[str]:
        return self._main_wrapper.obj_keys

    @property
    def runtime_key(self) -> str:
        return self._main_wrapper.runtime_key

    @property
    def n_actual_evals_in_opt(self) -> int:
        return self._main_wrapper._wrapper_vars.n_actual_evals_in_opt

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def get_results(self) -> dict[str, list[int | float | str | bool]]:
        return self._main_wrapper.get_results()

    def get_optimizer_overhead(self) -> dict[str, list[float]]:
        return self._main_wrapper.get_optimizer_overhead()

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        """
        Start the simulation using only the main process.
        Unlike parallel worker wrappers, each objective function will not run in parallel.
        Instead, we internally simulate the cumulative runtime for each worker.
        The optimizer must take the ask-and-tell interface.

        Args:
            opt (AbstractAskTellOptimizer):
                An optimizer that has `ask` and `tell` methods.
        """
        self._main_wrapper.simulate(opt)
