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
    * obj1_name -- List[obj1 at the n-th evaluation]
    * obj2_name -- List[obj2 at the n-th evaluation]
    :
    * objM_name -- List[objM at the n-th evaluation]
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

5. mfhpo-simulator-info/*/timestamp.json
    * worker1_id -- {"prev_timestamp": prev_timestamp1, "waited_time": waited_time1}
    * worker1_id -- {"prev_timestamp": prev_timestamp2, "waited_time": waited_time2}
    :
    * workerN_id -- {"prev_timestamp": prev_timestampN, "waited_time": waited_timeN}
This file tells the last checkpoint timestamp of each worker (prev_timestamp) and
how much time each worker waited for other workers in the last call.
"""
import os
import threading
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

from benchmark_simulator._constants import (
    DIR_NAME,
    INF,
    INIT_STATE,
    ObjectiveFuncType,
    PROC_ALLOC_NAME,
    _StateType,
    _get_file_paths,
)
from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _cache_state,
    _delete_state,
    _fetch_cache_states,
    _fetch_cumtimes,
    _fetch_timestamps,
    _init_simulator,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
    _record_timestamp,
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
        obj_func: ObjectiveFuncType,
        n_actual_evals_in_opt: int,
        n_evals: int,
        fidel_keys: Optional[List[str]] = None,
        obj_keys: List[str] = ["loss"][:],
        runtime_key: str = "runtime",
        seed: int = DEFAULT_SEED,
        continual_max_fidel: Optional[int] = None,
    ):
        """A worker class for each worker.
        This worker class is supposed to be instantiated for each worker.
        For example, if we use 4 workers for an optimization, then four instances should be created.

        Args:
            subdir_name (str):
                The subdirectory name to store all running information.
            n_workers (int):
                The number of workers to use. In other words, how many parallel workers to use.
            func (ObjectiveFuncType):
                A callable object that serves as the objective function.
                Args:
                    eval_config: Dict[str, Any]
                    fidels: Optional[Dict[str, Union[float, int]]]
                    seed: Optional[int]
                    **data_to_scatter: Any
                Returns:
                    results: Dict[str, float]
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
            fidel_keys (Optional[List[str]]):
                The fidelity names to be used in the objective function.
                If None, we assume that no fidelity is used.
            obj_keys (List[str]):
                The keys of the objective metrics used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            seed (int):
                The random seed to be used to allocate random seed to each call.
            continual_max_fidel (Optional[int]):
                The maximum fidelity to used in continual evaluations.
                This is valid only if we use a single fidelity.
                If not None, each call is a continuation from the call with the same eval_config and lower fidel.
                For example, when we already trained the objective with configA and training_epoch=10,
                we probably would like to continue the training from epoch 10 rather than from scratch
                for call with configA and training_epoch=30.
                continual_eval=True calculates the runtime considers this.
                If False, each call is considered to be processed from scratch.
        """
        self._guarantee_no_hang(n_workers=n_workers, n_actual_evals_in_opt=n_actual_evals_in_opt, n_evals=n_evals)
        self._worker_id = _generate_time_hash()
        self._dir_name = os.path.join(DIR_NAME, subdir_name)
        _, self._result_path, self._state_path, self._cumtime_path, self._timestamp_path = _get_file_paths(
            self.dir_name
        )
        self._init_worker()

        self._rng = np.random.RandomState(seed)
        self._use_fidel = fidel_keys is not None
        self._obj_func = obj_func
        self._max_fidel, self._n_evals = continual_max_fidel, n_evals
        self._obj_keys, self._runtime_key = obj_keys[:], runtime_key
        self._fidel_keys = [] if fidel_keys is None else fidel_keys[:]
        self._stored_obj_keys = list(set(self._obj_keys + [runtime_key]))
        self._index = self._alloc_index(n_workers)
        self._cumtime = 0.0
        self._continual_eval = continual_max_fidel is not None
        self._terminated = False
        self._validate_fidel_args()

    def __repr__(self) -> str:
        return f"Worker-{self._worker_id}"

    @property
    def dir_name(self) -> str:
        return self._dir_name

    def _guarantee_no_hang(self, n_workers: int, n_actual_evals_in_opt: int, n_evals: int) -> None:
        if n_actual_evals_in_opt < n_workers + n_evals:
            threshold = n_workers + n_evals
            # In fact, n_workers + n_evals - 1 is the real minimum threshold.
            raise ValueError(
                "Cannot guarantee that optimziers will not hang. "
                f"Use n_actual_evals_in_opt >= {threshold} (= n_evals + n_workers) at least. "
                "Note that our package cannot change your optimizer setting, so "
                "make sure that you changed your optimizer setting, but not only `n_actual_evals_in_opt`."
            )

    def _validate_fidel_args(self) -> None:
        # Guarantee the sufficiency: self._continual_eval ==> len(self._fidel_keys) == 1
        if self._continual_eval and len(self._fidel_keys) != 1:
            raise ValueError(
                f"continual_max_fidel is valid only if fidel_keys has only one element, but got {self._fidel_keys}"
            )

    def _init_worker(self) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        _init_simulator(dir_name=self.dir_name)
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=0.0)

    def _alloc_index(self, n_workers: int) -> int:
        worker_id_to_index = _wait_all_workers(path=self._cumtime_path, n_workers=n_workers)
        time.sleep(1e-2)  # buffer before the optimization
        return worker_id_to_index[self._worker_id]

    def _get_init_state(self) -> Tuple[_StateType, Optional[int]]:
        init_state = INIT_STATE[:]
        # initial seed, note: 1 << 30 is a huge number that fits 32bit.
        init_state[-1] = self._rng.randint(1 << 30)  # type: ignore
        return init_state, None

    def _get_cached_state_and_index(self, config_hash: int, fidel: int) -> Tuple[_StateType, Optional[int]]:
        # _StateType = List[_RuntimeType, _CumtimeType, _FidelityType, _SeedType]
        cached_states = _fetch_cache_states(self._state_path).get(config_hash, [])[:]
        intermediate_avail = [state[1] <= self._cumtime and state[2] < fidel for state in cached_states]
        cached_state_index = intermediate_avail.index(True) if any(intermediate_avail) else None
        if cached_state_index is None:
            return self._get_init_state()
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
        if fidel != self._max_fidel:  # update the cache data, TODO: Fix
            new_state = [total_runtime, self._cumtime, fidel, seed]
            _cache_state(new_state=new_state, update_index=cached_state_index, **kwargs)
        elif cached_state_index is not None:  # if None, newly start and train till the end, so no need to delete.
            _delete_state(index=cached_state_index, **kwargs)

    def _validate_output(self, results: Dict[str, float]) -> None:
        keys_in_output = set(results.keys())
        keys = set(self._stored_obj_keys)
        if keys_in_output.intersection(keys) != keys:
            raise KeyError(
                f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
                f"but got {results}"
            )

    def _proc_output_from_scratch(
        self, eval_config: Dict[str, Any], fidels: Optional[Dict[str, Union[float, int]]], **data_to_scatter: Any
    ) -> Dict[str, float]:
        _fidels: Dict[str, Union[float, int]] = {} if fidels is None else fidels.copy()
        if self._use_fidel and set(_fidels.keys()) != set(self._fidel_keys):
            raise KeyError(f"The keys in fidels must be identical to fidel_keys, but got {fidels}")

        seed = self._rng.randint(1 << 30)  # type: ignore
        results = self._obj_func(eval_config=eval_config, seed=seed, fidels=fidels, **data_to_scatter)
        self._validate_output(results)
        self._cumtime += results[self._runtime_key]
        return {k: results[k] for k in self._stored_obj_keys}

    def _proc_output_from_existing_state(
        self, eval_config: Dict[str, Any], fidel: int, **data_to_scatter: Any
    ) -> Dict[str, float]:
        config_hash: int = hash(str(eval_config))
        cached_state, cached_state_index = self._get_cached_state_and_index(config_hash=config_hash, fidel=fidel)
        cached_runtime, _, _, seed = cached_state
        _fidels: Dict[str, Union[float, int]] = {self._fidel_keys[0]: fidel}
        results = self._obj_func(eval_config=eval_config, seed=seed, fidels=_fidels, **data_to_scatter)
        self._validate_output(results)
        total_runtime = results[self._runtime_key]
        actual_runtime = max(0.0, total_runtime - cached_runtime)
        self._cumtime += actual_runtime
        self._update_state(
            total_runtime=total_runtime,
            cached_state_index=cached_state_index,
            seed=seed,
            config_hash=config_hash,
            fidel=fidel,
        )
        return {**{k: results[k] for k in self._obj_keys}, self._runtime_key: actual_runtime}

    def _proc_output(
        self, eval_config: Dict[str, Any], fidels: Optional[Dict[str, Union[float, int]]], **data_to_scatter: Any
    ) -> Dict[str, float]:
        if self._continual_eval:
            if fidels is None or len(fidels.values()) != 1:
                raise ValueError(
                    f"fidels must have only one element when continual_max_fidel is provided, but got {fidels}"
                )

            fidel = next(iter(fidels.values()))
            if not isinstance(fidel, int):
                raise ValueError(f"Fidelity for continual evaluation must be integer, but got {fidel}")

            return self._proc_output_from_existing_state(eval_config=eval_config, fidel=fidel, **data_to_scatter)
        else:
            return self._proc_output_from_scratch(eval_config=eval_config, fidels=fidels, **data_to_scatter)

    def _wait_other_workers(self) -> None:
        """
        Wait until the worker's cumulative runtime becomes the smallest.
        The smallest cumulative runtime implies that the order in the record will not disturbed
        even if the worker reports its results now.
        """
        wait_start = time.time()
        _wait_until_next(path=self._cumtime_path, worker_id=self._worker_id)
        _record_timestamp(
            path=self._timestamp_path,
            worker_id=self._worker_id,
            prev_timestamp=time.time(),
            waited_time=time.time() - wait_start,
        )

    def _post_proc(self, results: Dict[str, float]) -> None:
        # First, record the simulated cumulative runtime after calling the objective
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=self._cumtime)
        # Wait till the cumulative runtime becomes the smallest
        self._wait_other_workers()
        row = dict(cumtime=self._cumtime, index=self._index, **{k: results[k] for k in self._obj_keys})
        # Record the results to the main database when the cumulative runtime is the smallest
        _record_result(self._result_path, results=row)
        if _is_simulator_terminated(self._result_path, max_evals=self._n_evals):
            self._finish()

    def _load_timestamps(self) -> Tuple[float, float]:
        timestamp_dict = _fetch_timestamps(self._timestamp_path)
        if len(timestamp_dict) == 0:  # We do not need it right after the instantiation
            return 0.0, time.time()

        timestamp = timestamp_dict[self._worker_id]
        self._cumtime = _fetch_cumtimes(self._cumtime_path)[self._worker_id]
        self._terminated = self._cumtime >= INF - 1e-5  # INF means finish has been called.
        return timestamp["waited_time"], timestamp["prev_timestamp"]

    def __call__(
        self, eval_config: Dict[str, Any], fidels: Optional[Dict[str, Union[float, int]]] = None, **data_to_scatter: Any
    ) -> Dict[str, float]:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang at the end.

        Args:
            eval_config (Dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (Optional[Dict[str, Union[float, int]]]):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, no-fidelity opt.
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
        waited_time, prev_timestamp = self._load_timestamps()
        if self._terminated:
            return {**{k: INF for k in self._obj_keys}, self._runtime_key: INF}
        if not self._use_fidel and fidels is not None:
            raise ValueError(
                "Objective function got keyword `fidels`, but fidel_keys was not provided in worker instantiation."
            )
        if self._use_fidel and fidels is None:
            raise ValueError(
                "Objective function did not get keyword `fidels`, but fidel_keys was provided in worker instantiation."
            )

        sampling_time = max(0.0, time.time() - prev_timestamp - waited_time)
        self._cumtime += sampling_time

        results = self._proc_output(eval_config, fidels, **data_to_scatter)
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
        obj_func: ObjectiveFuncType,
        n_actual_evals_in_opt: int,
        n_evals: int,
        fidel_keys: Optional[List[str]] = None,
        obj_keys: List[str] = ["loss"][:],
        runtime_key: str = "runtime",
        seeds: Optional[List[int]] = None,
        continual_max_fidel: Optional[int] = None,
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
            obj_func (ObjectiveFuncType):
                A callable object that serves as the objective function.
                Args:
                    eval_config: Dict[str, Any]
                    fidels: Optional[Dict[str, Union[float, int]]]
                    seed: Optional[int]
                    **data_to_scatter: Any
                Returns:
                    results: Dict[str, float]
                        It must return `objective metric` and `runtime` at least.
            n_evals (int):
                How many configurations we evaluate.
                More specifically, how many times we call the objective function during the optimization.
            fidel_keys (Optional[List[str]]):
                The fidelity names to be used in the objective function.
                If None, we assume that no fidelity is used.
            obj_keys (List[str]):
                The keys of the objective metrics used in `results` returned by func.
            runtime_key (str):
                The key of the runtime metric used in `results` returned by func.
            continual_max_fidel (Optional[int]):
                The maximum fidelity to used in continual evaluations.
                This is valid only if we use a single fidelity.
                If not None, each call is a continuation from the call with the same eval_config and lower fidel.
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
            fidel_keys=fidel_keys,
            n_actual_evals_in_opt=n_actual_evals_in_opt,
            n_evals=n_evals,
            obj_keys=obj_keys[:],
            runtime_key=runtime_key,
            continual_max_fidel=continual_max_fidel,
        )
        self._obj_keys, self._runtime_key = obj_keys[:], runtime_key
        self._dir_name = os.path.join(DIR_NAME, subdir_name)
        self._n_workers = n_workers
        self._workers: List[ObjectiveFuncWorker]
        self._main_pid = os.getpid()
        self._init_workers(worker_kwargs, seeds=seeds)

        self._max_fidel = continual_max_fidel
        self._dir_name = self._workers[0].dir_name
        self._pid_to_index: Dict[int, int] = {}

    @property
    def dir_name(self) -> str:
        return self._dir_name

    def _init_workers(self, worker_kwargs: Dict[str, Any], seeds: Optional[List[int]]) -> None:
        seeds = [DEFAULT_SEED] * self._n_workers if seeds is None else seeds[:]
        if len(seeds) != self._n_workers:
            raise ValueError(f"The length of seeds must be n_workers={self._n_workers}, but got seeds={seeds}")
        if os.path.exists(self.dir_name):
            raise FileExistsError(f"The directory `{self.dir_name}` already exists. Remove it first.")

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

    def __call__(
        self, eval_config: Dict[str, Any], fidels: Optional[Dict[str, Union[float, int]]] = None, **data_to_scatter: Any
    ) -> Dict[str, float]:
        """The memta-wrapper method of the objective function method in WorkerFunc instances.

        This method recognizes each WorkerFunc by process ID and call the corresponding worker based on the ID.

        Args:
            eval_config (Dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (Optional[Dict[str, Union[float, int]]]):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, no-fidelity opt.
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
        pid = threading.get_ident() if pid == self._main_pid else pid
        if len(self._pid_to_index) != self._n_workers:
            self._init_alloc(pid)

        worker_index = self._pid_to_index[pid]
        results = self._workers[worker_index](eval_config=eval_config, fidels=fidels, **data_to_scatter)
        return results
