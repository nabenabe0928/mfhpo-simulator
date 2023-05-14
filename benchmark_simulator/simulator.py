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
    * loss -- List[loss at the n-th evaluation]
    * cumtime -- List[cumtime up to the n-th evaluation]
    * index -- List[the index of the worker of the n-th evaluation]
This file is necessary for post-hoc analysis.

3. mfhpo-simulator-info/*/state_cache.json
    * config1 -- List[Tuple[runtime, cumtime, fidel, seed]]
    * config2 -- List[Tuple[runtime, cumtime, fidel, seed]]
    :
    * configN -- List[Tuple[runtime, cumtime, fidel, seed]]
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
"""
import os
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

from benchmark_simulator._constants import (
    DIR_NAME,
    INF,
    INIT_STATE,
    PROC_ALLOC_NAME,
    _ObjectiveFunc,
    _StateType,
    _get_file_paths,
)
from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _cache_state,
    _delete_state,
    _fetch_cache_states,
    _init_simulator,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
    _wait_all_workers,
    _wait_proc_allocation,
    _wait_until_next,
)
from benchmark_simulator._utils import _generate_time_hash

import numpy as np


DEFAULT_SEED = 42


class ObjectiveFuncWorker:
    def __init__(
        self,
        subdir_name: str,
        n_workers: int,
        obj_func: _ObjectiveFunc,
        max_fidel: int,
        n_actual_evals_in_opt: int,
        max_evals: int,
        loss_key: str = "loss",
        runtime_key: str = "runtime",
        seed: int = DEFAULT_SEED,
        continual_eval: bool = True,
    ):
        """A worker class for each worker.
        This worker class is supposed to be instantiated for each worker.
        For example, if we use 4 workers for an optimization, then four instances should be created.

        Args:
            subdir_name (str):
                The subdirectory name to store all running information.
            n_workers (int):
                The number of workers to use. In other words, how many parallel workers to use.
            func (_ObjectiveFunc):
                A callable object that serves as the objective function.
                Args:
                    eval_config: Dict[str, Any]
                    fidel: int
                    seed: Optional[int]
                    **data_to_scatter: Any
                Returns:
                    results: Dict[str, float]
                        It must return `objective metric` and `runtime` at least.
            max_fidel (int):
                The maximum fidelity defined in the objective function.
            n_actual_evals_in_opt (int):
                The number of evaluations that optimizers do and it is used only for raising an error in init.
                Note that the number of evaluations means
                how many times we call the objective function during the optimization.
                This number is needed to automatically finish the worker class.
                We cannot know the timing of the termination without this information, and thus optimizers hang.
            max_evals (int):
                How many configurations we would like to collect.
                More specifically, how many times we call the objective function during the optimization.
                We can guarantee that `results.json` has at least this number of evaluations.
            loss_key (str):
                The key of the objective metric used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            seed (int):
                The random seed to be used to allocate random seed to each call.
            continual_eval (bool):
                Whether each call is a continuation from the call with the same eval_config and lower fidel.
                For example, when we already trained the objective with configA and training_epoch=10,
                we probably would like to continue the training from epoch 10 rather than from scratch
                for call with configA and training_epoch=30.
                continual_eval=True calculates the runtime considers this.
                If False, each call is considered to be processed from scratch.
        """
        self._worker_id = _generate_time_hash()
        self._dir_name = os.path.join(DIR_NAME, subdir_name)
        _, self._result_path, self._state_path, self._cumtime_path = _get_file_paths(self.dir_name)
        self._init_worker()

        self._rng = np.random.RandomState(seed)
        self._obj_func = obj_func
        self._max_fidel, self._max_evals = max_fidel, max_evals
        self._loss_key, self._runtime_key = loss_key, runtime_key
        self._index = self._alloc_index(n_workers)
        self._prev_timestamp, self._waited_time, self._cumtime = time.time(), 0.0, 0.0
        self._continual_eval = continual_eval
        self._terminated = False

    def __repr__(self) -> str:
        return f"Worker-{self._worker_id}"

    @property
    def dir_name(self) -> str:
        return self._dir_name

    def _guarantee_no_hang(self, n_workers: int, n_actual_evals_in_opt: int, max_evals: int) -> None:
        if n_actual_evals_in_opt < n_workers + max_evals:
            threshold = n_workers + max_evals
            # In fact, n_workers + max_evals - 1 is the real minimum threshold.
            raise ValueError(
                "Cannot guarantee that optimziers will not hang. "
                f"Use n_actual_evals_in_opt >= {threshold} (= max_evals + n_workers) at least. "
                "Note that our package cannot change your optimizer setting, so "
                "make sure that you changed your optimizer setting, but not only `n_actual_evals_in_opt`."
            )

    def _init_worker(self) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        _init_simulator(dir_name=self.dir_name)
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=0.0)

    def _alloc_index(self, n_workers: int) -> int:
        worker_id_to_index = _wait_all_workers(path=self._cumtime_path, n_workers=n_workers)
        time.sleep(1e-2)  # buffer before the optimization
        return worker_id_to_index[self._worker_id]

    def _get_cached_state_and_index(self, config_hash: int, fidel: int) -> Tuple[_StateType, Optional[int]]:
        # _StateType = List[_RuntimeType, _CumtimeType, _FidelityType, _SeedType]
        cached_states = _fetch_cache_states(self._state_path).get(config_hash, [])[:]
        intermediate_avail = [state[1] <= self._cumtime and state[2] < fidel for state in cached_states]
        cached_state_index = intermediate_avail.index(True) if any(intermediate_avail) else None
        if cached_state_index is None:
            init_state = INIT_STATE[:]
            # initial seed, note: 1 << 30 is a huge number that fits 32bit.
            init_state[-1] = self._rng.randint(1 << 30)  # type: ignore
            return init_state, None
        else:
            return cached_states[cached_state_index][:], cached_state_index

    def _update_state(
        self,
        config_hash: int,
        fidel: int,
        total_runtime: float,
        seed: Optional[int],
        cached_state_index: Optional[int],
    ) -> None:
        kwargs = dict(path=self._state_path, config_hash=config_hash)
        if fidel != self._max_fidel:  # update the cache data
            new_state = [total_runtime, self._cumtime, fidel, seed]
            _cache_state(new_state=new_state, update_index=cached_state_index, **kwargs)
        elif cached_state_index is not None:  # if None, newly start and train till the end, so no need to delete.
            _delete_state(index=cached_state_index, **kwargs)

    def _proc_output(self, eval_config: Dict[str, Any], fidel: int, **data_to_scatter: Any) -> Dict[str, float]:
        config_hash = hash(str(eval_config))
        kwargs = dict(config_hash=config_hash, fidel=fidel)
        cached_state, cached_state_index = self._get_cached_state_and_index(**kwargs)
        cached_runtime, _, _, seed = cached_state
        results = self._obj_func(eval_config=eval_config, fidel=fidel, seed=seed, **data_to_scatter)
        loss, total_runtime = results[self._loss_key], results[self._runtime_key]
        actual_runtime = max(0.0, total_runtime - cached_runtime) if self._continual_eval else total_runtime
        self._cumtime += actual_runtime
        self._update_state(total_runtime=total_runtime, cached_state_index=cached_state_index, seed=seed, **kwargs)
        return {self._loss_key: loss, self._runtime_key: actual_runtime}

    def _wait_other_workers(self) -> None:
        """
        Wait until the worker's cumulative runtime becomes the smallest.
        The smallest cumulative runtime implies that the order in the record will not disturbed
        even if the worker reports its results now.
        """
        wait_start = time.time()
        _wait_until_next(path=self._cumtime_path, worker_id=self._worker_id)
        self._waited_time = time.time() - wait_start
        self._prev_timestamp = time.time()

    def _post_proc(self, results: Dict[str, float]) -> None:
        # First, record the simulated cumulative runtime after calling the objective
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=self._cumtime)
        # Wait till the cumulative runtime becomes the smallest
        self._wait_other_workers()
        row = dict(loss=results[self._loss_key], cumtime=self._cumtime, index=self._index)
        # Record the results to the main database when the cumulative runtime is the smallest
        _record_result(self._result_path, results=row)
        if _is_simulator_terminated(self._result_path, max_evals=self._max_evals):
            self._finish()

    def __call__(self, eval_config: Dict[str, Any], fidel: int, **data_to_scatter: Any) -> Dict[str, float]:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang at the end.

        Args:
            eval_config (Dict[str, Any]):
                The configuration to be used in the objective function.
            fidel (int):
                The fidelity to be used in the objective function. Typically training epoch in deep learning.
            **data_to_scatter (Any):
                Data to scatter across workers.
                For example, when the objective function instance has a large file,
                Dask, which is a typical module for parallel optimization, must serialize/deserialize
                the objective function instances. It causes a significant bottleneck.
                By using dask.scatter, we can avoid this problem and this kwargs serves for this purpose.
                Note that since the handling of parallel workers vary depending on packages,
                users must adapt by themselves.

        Returns:
            results (Dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        if self._terminated:
            return {self._loss_key: INF, self._runtime_key: INF}

        sampling_time = max(0.0, time.time() - self._prev_timestamp - self._waited_time)
        self._cumtime += sampling_time

        results = self._proc_output(eval_config, fidel, **data_to_scatter)
        self._post_proc(results)
        return results

    def _finish(self) -> None:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang.
        """
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=INF)
        self._terminated = True


class CentralWorkerManager:
    def __init__(
        self,
        subdir_name: str,
        n_workers: int,
        obj_func: _ObjectiveFunc,
        max_fidel: int,
        n_actual_evals_in_opt: int,
        max_evals: int,
        loss_key: str = "loss",
        runtime_key: str = "runtime",
        seeds: Optional[List[int]] = None,
        continual_eval: bool = True,
    ):
        """A central worker manager class.
        This class is supposed to be instantiated if the optimizer module uses multiprocessing.
        For example, Dask, multiprocessing, and joblib would need this class.
        This class recognizes each worker by process ID.
        Therefore, process ID for each worker must be always unique and identical.

        Args:
            subdir_name (str):
                The subdirectory name to store all running information.
            n_workers (int):
                The number of workers to use. In other words, how many parallel workers to use.
            obj_func (_ObjectiveFunc):
                A callable object that serves as the objective function.
                Args:
                    eval_config: Dict[str, Any]
                    fidel: int
                    seed: Optional[int]
                    **data_to_scatter: Any
                Returns:
                    results: Dict[str, float]
                        It must return `objective metric` and `runtime` at least.
            max_fidel (int):
                The maximum fidelity defined in the objective function.
            max_evals (int):
                How many configurations we evaluate.
                More specifically, how many times we call the objective function during the optimization.
            loss_key (str):
                The key of the objective metric used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            continual_eval (bool):
                Whether each call is a continuation from the call with the same eval_config and lower fidel.
                For example, when we already trained the objective with configA and training_epoch=10,
                we probably would like to continue the training from epoch 10 rather than from scratch
                for call with configA and training_epoch=30.
                continual_eval=True calculates the runtime considers this.
                If False, each call is considered to be processed from scratch.
        """
        worker_kwargs = dict(
            obj_func=obj_func,
            n_workers=n_workers,
            subdir_name=subdir_name,
            max_fidel=max_fidel,
            n_actual_evals_in_opt=n_actual_evals_in_opt,
            max_evals=max_evals,
            loss_key=loss_key,
            runtime_key=runtime_key,
            continual_eval=continual_eval,
        )
        self._n_workers = n_workers
        self._workers: List[ObjectiveFuncWorker]
        self._init_workers(worker_kwargs, seeds=seeds)

        self._dir_name = self._workers[0].dir_name
        self._pid_to_index: Dict[int, int] = {}

    def _init_workers(self, worker_kwargs: Dict[str, Any], seeds: Optional[List[int]]) -> None:
        seeds = [DEFAULT_SEED] * self._n_workers if seeds is None else seeds[:]
        if len(seeds) != self._n_workers:
            raise ValueError(f"The length of seeds must be n_workers={self._n_workers}, but got seeds={seeds}")

        pool = Pool()
        results = []
        for _, seed in enumerate(seeds):
            results.append(pool.apply_async(ObjectiveFuncWorker, kwds=dict(**worker_kwargs, seed=seed)))

        pool.close()
        pool.join()
        self._workers = [result.get() for result in results]

    def _init_alloc(self, pid: int) -> None:
        _path = os.path.join(self._dir_name, PROC_ALLOC_NAME)
        _allocate_proc_to_worker(path=_path, pid=pid)
        self._pid_to_index = _wait_proc_allocation(path=_path, n_workers=self._n_workers)

    def __call__(self, eval_config: Dict[str, Any], fidel: int, **data_to_scatter: Any) -> Dict[str, float]:
        """The memta-wrapper method of the objective function method in WorkerFunc instances.

        This method recognizes each WorkerFunc by process ID and call the corresponding worker based on the ID.

        Args:
            eval_config (Dict[str, Any]):
                The configuration to be used in the objective function.
            fidel (int):
                The fidelity to be used in the objective function. Typically training epoch in deep learning.
            **data_to_scatter (Any):
                Data to scatter across workers.
                For example, when the objective function instance has a large file,
                Dask, which is a typical module for parallel optimization, must serialize/deserialize
                the objective function instances. It causes a significant bottleneck.
                By using dask.scatter, we can avoid this problem and this kwargs serves for this purpose.
                Note that since the handling of parallel workers vary depending on packages,
                users must adapt by themselves.

        Returns:
            results (Dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        pid = os.getpid()
        if len(self._pid_to_index) != self._n_workers:
            self._init_alloc(pid)

        worker_index = self._pid_to_index[pid]
        results = self._workers[worker_index](eval_config=eval_config, fidel=fidel, **data_to_scatter)
        return results
