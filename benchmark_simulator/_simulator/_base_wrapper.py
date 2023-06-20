from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod

from benchmark_simulator._constants import DIR_NAME, ObjectiveFuncType, _WrapperVars, _get_file_paths
from benchmark_simulator._utils import _SecureLock

import numpy as np


class _BaseWrapperInterface(metaclass=ABCMeta):
    """A base wrapper class for each worker or manager.
    This wrapper class serves the shared interface of worker and manager class.

    Attributes:
        dir_name (str):
            The directory name where all the information will be stored.
        obj_keys (list[str]):
            The objective (or constraint) names that will be stored in the result file.
        runtime_key (str):
            The runtime name that will be used for the scheduling.
        fidel_keys (list[str]):
            The fidelity names that will be used in the input `fidels`.
    """

    def __init__(
        self,
        subdir_name: str,
        n_workers: int,
        obj_func: ObjectiveFuncType,
        n_actual_evals_in_opt: int,
        n_evals: int,
        fidel_keys: list[str] | None = None,
        obj_keys: list[str] | None = None,
        runtime_key: str = "runtime",
        seed: int | None = None,
        continual_max_fidel: int | None = None,
        max_waiting_time: float = np.inf,
        check_interval_time: float = 1e-4,
        store_config: bool = False,
    ):
        """The initialization of a wrapper class.

        Both ObjectiveFuncWorker and CentralWorkerManager have the same interface and the same set of control params.

        Args:
            subdir_name (str):
                The subdirectory name to store all running information.
            n_workers (int):
                The number of workers to use. In other words, how many parallel workers to use.
            obj_func (ObjectiveFuncType):
                A callable object that serves as the objective function.
                Args:
                    eval_config: dict[str, Any]
                    fidels: dict[str, int | float] | None
                    seed: int | None
                    **data_to_scatter: Any
                Returns:
                    results: dict[str, float]
                        It must return `objective metric` and `runtime` at least.
            n_actual_evals_in_opt (int):
                The number of evaluations that optimizers do and it is used only for raising an error in init.
                Note that the number of evaluations means
                how many times we call the objective function during the optimization.
                This number is needed to automatically finish the worker class.
                We cannot know the timing of the termination without this information, and thus optimizers hang.
            n_evals (int):
                How many configurations we would like to collect.
                More specifically, how many times we call the objective function during the optimization.
                We can guarantee that `results.json` has at least this number of evaluations.
            fidel_keys (list[str] | None):
                The fidelity names to be used in the objective function.
                If None, we assume that no fidelity is used.
            obj_keys (list[str] | None):
                The keys of the objective metrics used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            seed (int | None):
                The random seed to be used to allocate random seed to each call.
            continual_max_fidel (int | None):
                The maximum fidelity to used in continual evaluations.
                This is valid only if we use a single fidelity.
                If not None, each call is a continuation from the call with the same eval_config and lower fidel.
                For example, when we already trained the objective with configA and training_epoch=10,
                we probably would like to continue the training from epoch 10 rather than from scratch
                for call with configA and training_epoch=30.
                continual_eval=True calculates the runtime considers this.
                If False, each call is considered to be processed from scratch.
            check_interval_time (float):
                How often each worker should check whether they could be assigned a new job.
                For example, if 1e-2 is specified, each worker check whether they can get a new job every 1e-2 seconds.
                If there are many workers, too small check_interval_time may cause a big bottleneck.
                On the other hand, a big check_interval_time spends more time for waiting.
                By default, check_interval_time is set to a relatively small number, so users might rather want to
                increase the number to avoid the bottleneck for many workers.
            max_waiting_time (float):
                The maximum waiting time to judge hang.
                If any one of the workers does not do any updates for this amount of time, we raise TimeoutError.
            store_config (bool):
                Whether to store all config/fidel information.
                The information is sorted chronologically.
                When you do large-scale experiments, this may incur too much storage consumption.
        """
        self._wrapper_vars = _WrapperVars(
            subdir_name=subdir_name,
            n_workers=n_workers,
            obj_func=obj_func,
            n_actual_evals_in_opt=n_actual_evals_in_opt,
            n_evals=n_evals,
            fidel_keys=fidel_keys,
            obj_keys=obj_keys if obj_keys is not None else ["loss"],
            runtime_key=runtime_key,
            seed=seed,
            continual_max_fidel=continual_max_fidel,
            max_waiting_time=max_waiting_time,
            check_interval_time=check_interval_time,
            store_config=store_config,
        )
        self._lock = _SecureLock()
        self._dir_name = os.path.join(DIR_NAME, subdir_name)
        self._paths = _get_file_paths(self.dir_name)
        self._obj_keys, self._runtime_key = self._wrapper_vars.obj_keys, runtime_key
        self._fidel_keys = [] if fidel_keys is None else fidel_keys[:]
        self._init_wrapper()

    @property
    def dir_name(self) -> str:
        return self._dir_name

    @property
    def obj_keys(self) -> list[str]:
        return self._obj_keys[:]

    @property
    def runtime_key(self) -> str:
        return self._runtime_key

    @property
    def fidel_keys(self) -> list[str]:
        return self._fidel_keys[:]

    @abstractmethod
    def _init_wrapper(self) -> None:
        raise NotImplementedError
