from __future__ import annotations

from typing import TYPE_CHECKING

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
        n_evals: int = 100,
        store_actual_cumtime: bool = False,
        allow_parallel_sampling: bool = False,
        max_total_eval_time: float = float("inf"),
        expensive_sampler: bool = False,
    ):
        """The initialization of a wrapper class for ask-and-tell optimization.

        Args:
            obj_func (ObjectiveFuncType):
                A callable object that serves as the objective function.
                Args:
                    eval_config: dict[str, Any]
                Returns:
                    results: list[float]
                        The last element must be the runtime.
                        All preceding elements are objective metrics.
            n_workers (int):
                The number of simulated workers. In other words, how many parallel workers to simulate.
            n_evals (int):
                How many configurations we would like to collect.
            store_actual_cumtime (bool):
                Whether to store actual cumulative time at each point.
            allow_parallel_sampling (bool):
                Whether sampling can happen in parallel.
            max_total_eval_time (float):
                The maximum total evaluation time for the optimization.
            expensive_sampler (bool):
                Whether the optimizer is expensive relative to a function evaluation.
        """
        self._n_workers = n_workers
        wrapper_vars = _WrapperVars(
            obj_func=obj_func,
            n_workers=n_workers,
            n_evals=n_evals,
            max_total_eval_time=max_total_eval_time,
            store_actual_cumtime=store_actual_cumtime,
            allow_parallel_sampling=allow_parallel_sampling,
            expensive_sampler=expensive_sampler,
        )

        self._main_wrapper = _AskTellWorkerManager(wrapper_vars)

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
