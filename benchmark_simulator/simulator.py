""" A collection of utilities for a simulation of multi-fidelity optimizations.

In multi-fidelity optimizations, as we often evaluate configs in parallel,
it is essential to manage the order of config allocations to each worker.
We allow such appropriate allocations with the utilities in this file.
To guarantee the safety, we share variables with each process using the file system.

We first define some terminologies:
* simulator: A system that manages simulated runtimes to allow optimizations of a tabular/surrogate benchmark
             without waiting for actual runtimes.
* worker: Worker instantiates an objective function with a given config and evaluates it.
* proc: Each worker will be instantiated in each process.
* worker_id: The time hash generated for each worker.
* pid or proc_id: The process ID for each proc obtained by `os.getpid()`, which is an integer.
* index: The index of each worker. (it will be one of the integers in [0, N - 1] where N is the # of workers.)

Now we describe each file shared with each process.
Note that each file takes the json-dict format and we write down as follows:
    * key1 -- val1
    * key2 -- val2
    :
    * keyN -- valN

1. mfhpo-simulator-info/*/proc_alloc.json
    * proc1_id -- worker1_id
    * proc2_id -- worker2_id
    :
    * procN_id -- workerN_id
Note that we need this file only if we have multiple processes in one run.
For example, when we use multiprocessing, we might need it.
`procX_id` is fetched by `os.getpid()` and `workerX_id` is initially generated by `generate_time_hash()`.

2. mfhpo-simulator-info/*/results.json
    * obj1_name -- list[obj1 at the n-th evaluation]
    * obj2_name -- list[obj2 at the n-th evaluation]
    :
    * objM_name -- list[objM at the n-th evaluation]
    * cumtime -- list[cumtime up to the n-th evaluation]
    * worker_index -- list[the index of the worker of the n-th evaluation]
This file is necessary for post-hoc analysis.

3. mfhpo-simulator-info/*/state_cache.json
    * config1 -- list[tuple[runtime, cumtime, fidel, seed]]
    * config2 -- list[tuple[runtime, cumtime, fidel, seed]]
    :
    * configN -- list[tuple[runtime, cumtime, fidel, seed]]
This file tells you the states of each config.
Runtime tells how long it took to evaluate configX up to the intermediate result.
Since we would like to use this information only for the restart of trainings,
we discard the information after each config reaches the full-fidelity training.
Each list gets more than two elements if evaluations of the same configs happen.

4. mfhpo-simulator-info/*/simulated_cumtime.json
    * worker1_id -- cumtime1
    * worker2_id -- cumtime2
    :
    * workerN_id -- cumtimeN
This file tells you how much time each worker virtually spends in the simulation
and we need this information to manage the order of job allocations to each worker.

5. mfhpo-simulator-info/*/timestamp.json
    * worker1_id -- prev_timestamp1
    * worker2_id -- prev_timestamp2
    :
    * workerN_id -- prev_timestampN
This file tells the last checkpoint timestamp of each worker (prev_timestamp).

6. mfhpo-simulator-info/*/sampled_time.json
    * before_sample -- [time1 before sample, time2 before sample, ...]
    * after_sample -- [time1 after sample, time2 after sample, ...]
    * worker_index -- [worker_index of the 1st eval, worker_index of the 2nd eval, ...]
This file is used to consider the sampling time.
after_sample is the latest cumtime immediately after the last sample and before_sample is before the last sample.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from benchmark_simulator._constants import AbstractAskTellOptimizer, DIR_NAME, ObjectiveFuncType, _WrapperVars
from benchmark_simulator._simulator._worker import _ObjectiveFuncWorker
from benchmark_simulator._simulator._worker_manager import _CentralWorkerManager
from benchmark_simulator._simulator._worker_manager_for_ask_and_tell import _AskTellWorkerManager

import numpy as np


def get_multiple_wrappers(
    obj_func: ObjectiveFuncType,
    save_dir_name: str | None = None,
    n_workers: int = 4,
    n_actual_evals_in_opt: int = 105,
    n_evals: int = 100,
    fidel_keys: list[str] | None = None,
    obj_keys: list[str] | None = None,
    runtime_key: str = "runtime",
    seed: int | None = None,
    continual_max_fidel: int | None = None,
    max_waiting_time: float = np.inf,
    check_interval_time: float = 1e-4,
    store_config: bool = False,
    allow_parallel_sampling: bool = False,
    config_tracking: bool = True,
    max_total_eval_time: float = np.inf,
) -> list[ObjectiveFuncWrapper]:
    """Return multiple wrapper instances.

    Args:
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
        save_dir_name (str | None):
            The subdirectory name to store all running information.
        n_workers (int):
            The number of workers to use. In other words, how many parallel workers to use.
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
        allow_parallel_sampling (bool):
            Whether sampling can happen in parallel.
            In many cases, sampler will not be run in parallel and then allow_parallel_sampling should be False.
            The default value is False.
        config_tracking (bool):
            Whether to validate config_id provided from the user side.
            It slows the simulation down when n_evals is large (> 3000),
            but it is recommended to avoid unexpected bugs that could happen.
        max_total_eval_time (float):
            The maximum total evaluation time for the optimization.
            For example, if max_total_eval_time=3600, the simulation evaluates until the simulated cumulative time
            reaches 3600 seconds.
            It is useful to combine with a large n_evals and n_actual_evals_in_opt.

    Returns:
        wrappers (list[ObjectiveFuncWrapper]):
            A list of wrappers.
            It contains n_workers of worker wrappers.
    """
    curtime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    save_dir_name = save_dir_name if save_dir_name is not None else f"data-{curtime}"

    dir_name = os.path.join(DIR_NAME, save_dir_name)
    if os.path.exists(dir_name):
        raise FileExistsError(f"The directory `{dir_name}` already exists. Remove it first.")

    wrapper_kwargs = dict(
        obj_func=obj_func,
        save_dir_name=save_dir_name,
        launch_multiple_wrappers_from_user_side=True,
        n_workers=n_workers,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        fidel_keys=fidel_keys,
        obj_keys=obj_keys,
        runtime_key=runtime_key,
        seed=seed,
        continual_max_fidel=continual_max_fidel,
        max_waiting_time=max_waiting_time,
        check_interval_time=check_interval_time,
        store_config=store_config,
        allow_parallel_sampling=allow_parallel_sampling,
        config_tracking=config_tracking,
        max_total_eval_time=max_total_eval_time,
        _async_instantiations=False,
    )
    return [ObjectiveFuncWrapper(**wrapper_kwargs, worker_index=i) for i in range(n_workers)]  # type: ignore[arg-type]


class ObjectiveFuncWrapper:
    """Objective function wrapper API for users.

    Please check more details at https://github.com/nabenabe0928/mfhpo-simulator/

    Attributes:
        dir_name (str):
            The relative path where results will be stored.
            In principle, it returns `./mfhpo-simulator/<save_dir_name>`.
        obj_keys (list[str]):
            The objective names that will be collected in results.
            The returned dict from users' objective functions must contain these keys.
            If you want to include the runtime in the results, you also need to include the runtime_key in obj_keys.
        runtime_key (str):
            The runtime name that will be used to define the runtime which the user objective function really took.
            The returned dict from users' objective functions must contain this key.
        fidel_keys (list[str]):
            The fidelity names that will be used in users' objective functions.
            `fidels` passed to the objective functions must contain these keys.
            When `continual_max_fidel=True`, fidel_keys can contain only one key and this fidelity will be used for
            the definition of continual learning.
        n_actual_evals_in_opt (int):
            The number of configuration evaluations during the actual optimization.
            Note that even if some configurations are continuations from an existing config with lower fidelity,
            they are counted as separated config evaluations.
        n_workers (int):
            The number of workers used in the user-defined optimizer.

    Methods:
        __call__(...) -> dict[str, float]:
            The wrapped objective function used in the user-defined optimizer. Valid only if `ask_and_tell=False`.
            Please check `ObjectiveFuncWrapper.__call__.__doc__` for more details.
        simulate(opt: AbstractAskTellOptimizer) -> None:
            The optimization loop for the wrapped objective function and the user-defined optimizer.
            Valid only if `ask_and_tell=True`.
            Please check `ObjectiveFuncWrapper.simulate.__doc__` for more details.
    """

    def __init__(
        self,
        obj_func: ObjectiveFuncType,
        launch_multiple_wrappers_from_user_side: bool = False,
        ask_and_tell: bool = False,
        save_dir_name: str | None = None,
        n_workers: int = 4,
        n_actual_evals_in_opt: int = 105,
        n_evals: int = 100,
        fidel_keys: list[str] | None = None,
        obj_keys: list[str] | None = None,
        runtime_key: str = "runtime",
        seed: int | None = None,
        continual_max_fidel: int | None = None,
        max_waiting_time: float = np.inf,
        check_interval_time: float = 1e-4,
        store_config: bool = False,
        allow_parallel_sampling: bool = False,
        config_tracking: bool = True,
        worker_index: int | None = None,
        max_total_eval_time: float = np.inf,
        careful_init: bool = False,
        _async_instantiations: bool = True,
    ):
        """The initialization of a wrapper class.

        Both ObjectiveFuncWorker and CentralWorkerManager have the same interface and the same set of control params.

        Args:
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
            launch_multiple_wrappers_from_user_side (bool):
                Whether users need to launch multiple objective function wrappers from user side.
                Examples of such cases are available at:
                    - https://github.com/nabenabe0928/mfhpo-simulator/blob/main/examples/bohb.py
                    - https://github.com/nabenabe0928/mfhpo-simulator/blob/main/examples/neps.py
                The first case is so obvious that users need to instantiate multiple worker objects.
                The second case is a bit tricky because multiple workers are instantiated in different main processes.
                In this case as well, it is obviously not one worker launch.
            ask_and_tell (bool):
                Whether to use an ask-and-tell interface optimizer.
                If True, the optimization loop will be run in the API side and hence users need to call simulate()
                to start simulation and the wrapper will be an optimizer wrapper rather than a function wrapper.
            save_dir_name (str | None):
                The subdirectory name to store all running information.
            n_workers (int):
                The number of workers to use. In other words, how many parallel workers to use.
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
            allow_parallel_sampling (bool):
                Whether sampling can happen in parallel.
                In many cases, sampler will not be run in parallel and then allow_parallel_sampling should be False.
                The default value is False.
            config_tracking (bool):
                Whether to validate config_id provided from the user side.
                It slows the simulation down when n_evals is large (> 3000),
                but it is recommended to avoid unexpected bugs that could happen.
            worker_index (int | None):
                It specifies which worker index will be used for this wrapper.
                It is typically useful when you run this wrapper from different processes in parallel.
                If you did not specify this index, our wrapper automatically allocates worker indices,
                but this may sometimes fail (in our environment with 0.01% of the probability for n_workers=8).
                The failure rate might be higher especially when you use a large n_workers, so in that case,
                probably users would like to use this option.
                The worker indices must be unique across the parallel workers and must be in [0, n_workers - 1].
            max_total_eval_time (float):
                The maximum total evaluation time for the optimization.
                For example, if max_total_eval_time=3600, the simulation evaluates until the simulated cumulative time
                reaches 3600 seconds.
                It is useful to combine with a large n_evals and n_actual_evals_in_opt.
            careful_init (bool):
                Whether doing initialization very carefully or not in the default setup (and only for the default).
                If True, we try to match the initialization order using sleep.
                It is not necessary for normal usage, but if users expect perfect reproducibility, users want to use it.
            _async_instantiations (bool):
                Whether each worker is instantiated asynchrously.
                In other words, whether to wait all workers' instantiations or not.
                This argument must not be touched by users.
        """
        curtime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        wrapper_vars = _WrapperVars(
            obj_func=obj_func,
            save_dir_name=save_dir_name if save_dir_name is not None else f"data-{curtime}",
            n_workers=n_workers,
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
            allow_parallel_sampling=allow_parallel_sampling,
            config_tracking=config_tracking,
            max_total_eval_time=max_total_eval_time,
        )

        self._main_wrapper: _AskTellWorkerManager | _CentralWorkerManager | _ObjectiveFuncWorker
        self._validate(
            save_dir_name=save_dir_name,
            ask_and_tell=ask_and_tell,
            launch_multiple_wrappers_from_user_side=launch_multiple_wrappers_from_user_side,
            worker_index=worker_index,
            n_workers=n_workers,
        )
        if ask_and_tell:
            self._main_wrapper = _AskTellWorkerManager(wrapper_vars)
        elif launch_multiple_wrappers_from_user_side:
            self._main_wrapper = _ObjectiveFuncWorker(
                wrapper_vars, worker_index=worker_index, async_instantiations=_async_instantiations
            )
        else:
            self._main_wrapper = _CentralWorkerManager(wrapper_vars, careful_init=careful_init)

    @property
    def dir_name(self) -> str:
        return self._main_wrapper.dir_name

    @property
    def result_file_path(self) -> str:
        return self._main_wrapper._paths.result

    @property
    def optimizer_overhead_file_path(self) -> str:
        return self._main_wrapper._paths.sampled_time

    @property
    def obj_keys(self) -> list[str]:
        return self._main_wrapper.obj_keys

    @property
    def runtime_key(self) -> str:
        return self._main_wrapper.runtime_key

    @property
    def fidel_keys(self) -> list[str]:
        return self._main_wrapper.fidel_keys

    @property
    def n_actual_evals_in_opt(self) -> int:
        return self._main_wrapper._wrapper_vars.n_actual_evals_in_opt

    @property
    def n_workers(self) -> int:
        return self._main_wrapper._wrapper_vars.n_workers

    def get_results(self) -> dict[str, list[int | float | str | bool]]:
        with open(self.result_file_path, mode="r") as f:
            results = json.load(f)

        return results

    def get_optimizer_overhead(self) -> dict[str, list[float]]:
        with open(self.optimizer_overhead_file_path, mode="r") as f:
            results = json.load(f)

        return results

    def _validate(
        self,
        save_dir_name: str | None,
        ask_and_tell: bool,
        launch_multiple_wrappers_from_user_side: bool,
        n_workers: int,
        worker_index: int | None,
    ) -> None:
        if ask_and_tell and launch_multiple_wrappers_from_user_side:
            raise ValueError(
                "ask_and_tell and launch_multiple_wrappers_from_user_side cannot be True at the same time."
            )
        if launch_multiple_wrappers_from_user_side and save_dir_name is None:
            raise ValueError(
                "When launch_multiple_wrappers_from_user_side is False, save_dir_name must be specified so that \n"
                "each worker recognizes with which processes it shares the optimization results."
            )
        if worker_index is not None and (ask_and_tell or not launch_multiple_wrappers_from_user_side):
            raise ValueError(
                "When launch_multiple_wrappers_from_user_side=False or ask_and_tell=True, "
                "worker_index cannot be specified."
            )
        if worker_index is not None and worker_index not in list(range(n_workers)):
            raise ValueError(f"worker_index must be in [0, {n_workers-1=}], but got {worker_index=}")

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        """The meta-wrapper method of the objective function method in WorkerFunc instances.

        This method recognizes each WorkerFunc by process ID and call the corresponding worker based on the ID.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the user-defined objective function.
                This configuration will not be necessary for the processing in our API, but the storage purpose.
                However, it will be directly passed to the user-defined objective function, hence we need it.
            fidels (dict[str, int | float] | None):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, no-fidelity opt.
            config_id (int | None):
                The identifier of configuration if needed for continual learning.
                As we internally use a hash of eval_config, it may be unstable if eval_config has float.
                However, even if config_id is not provided, our simulator works without errors
                although we cannot guarantee that our simulator recognizes the same configs if a users' optimizer
                slightly changes the content of eval_config.
            **data_to_scatter (Any):
                Data to scatter across workers.
                Users can pass any necessary information to the objective function,
                but this variable will NOT be used by our API at all.
                For example, when the objective function instance has a large file,
                Dask, which is a typical module for parallel optimization, must serialize/deserialize
                the objective function instances. It causes a significant bottleneck.
                By using dask.scatter, we can avoid this problem and this kwargs serves for this purpose.
                Note that since the handling of parallel workers vary depending on packages,
                users must adapt by themselves.

        Returns:
            results (dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        return self._main_wrapper(eval_config=eval_config, fidels=fidels, config_id=config_id, **data_to_scatter)

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        """
        Start the simulation using only the main process.
        Unlike the other worker wrappers, each objective function will not run in parallel.
        Instead, we internally simulate the cumulative runtime for each worker.
        For this sake, the optimizer must take so-called ask-and-tell interface.
        It means that optimizer can communicate with this class via `ask` and `tell` methods.
        As long as the optimizer takes this interface, arbitrary optimizers can be used for this class.

        Although this class may not be able to guarantee the exact behavior using parallel optimization,
        this class is safer than the other wrappers because it is thread-safe.
        Furthermore, if users want to try a large n_workers, this class is much safer and executable.

        Args:
            opt (AbstractAskTellOptimizer):
                An optimizer that has `ask` and `tell` methods.
                For example, if we run a sequential optimization, the expected loop looks like:
                    for i in range(100):
                        eval_config, fidels = opt.ask()
                        results = obj_func(eval_config, fidels)
                        opt.tell(eval_config, results, fidels)
        """
        self._main_wrapper.simulate(opt)
