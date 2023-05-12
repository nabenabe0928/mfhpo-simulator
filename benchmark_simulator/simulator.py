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
    * config1 -- List[Tuple[runtime, cumtime, budget, seed]]
    * config2 -- List[Tuple[runtime, cumtime, budget, seed]]
    :
    * configN -- List[Tuple[runtime, cumtime, budget, seed]]
This file tells you the states of each config.
Runtime tells how long it took to evaluate configX up to the intermediate result.
Since we would like to use this information only for the restart of trainings,
we discard the information after each config reaches the full-budget training.
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
    RESULT_FILE_NAME,
    STATE_CACHE_FILE_NAME,
    WORKER_CUMTIME_FILE_NAME,
    _ObjectiveFunc,
    _StateType,
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


class WorkerFunc:
    def __init__(
        self,
        subdir_name: str,
        n_workers: int,
        func: _ObjectiveFunc,
        max_budget: int,
        loss_key: str = "loss",
        runtime_key: str = "runtime",
    ):
        worker_id = _generate_time_hash()
        self._dir_name = os.path.join(DIR_NAME, subdir_name)
        os.makedirs(self.dir_name, exist_ok=True)
        _init_simulator(dir_name=self.dir_name)
        self._cumtime_path = os.path.join(self.dir_name, WORKER_CUMTIME_FILE_NAME)
        _record_cumtime(path=self._cumtime_path, worker_id=worker_id, cumtime=0.0)

        self._rng = np.random.RandomState(42)
        self._func = func
        self._result_path = os.path.join(self._dir_name, RESULT_FILE_NAME)
        self._state_path = os.path.join(self.dir_name, STATE_CACHE_FILE_NAME)
        self._max_budget = max_budget
        self._runtime_key = runtime_key
        self._loss_key = loss_key
        self._terminated = False
        self._worker_id = worker_id
        self._worker_id_to_index = _wait_all_workers(path=self._cumtime_path, n_workers=n_workers)
        time.sleep(1e-2)  # buffer before the optimization
        self._index = self._worker_id_to_index[self._worker_id]
        self._prev_timestamp = time.time()
        self._cumtime = 0.0

    def __repr__(self) -> str:
        return f"Worker-{self._worker_id}"

    @property
    def dir_name(self) -> str:
        return self._dir_name

    def _get_cached_state_and_index(self, config_hash: int, budget: int) -> Tuple[_StateType, Optional[int]]:
        # _StateType = List[_RuntimeType, _CumtimeType, _BudgetType, _SeedType]
        cached_states = _fetch_cache_states(self._state_path).get(config_hash, [])[:]
        intermediate_avail = [state[1] <= self._cumtime and state[2] < budget for state in cached_states]
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
        budget: int,
        total_runtime: float,
        seed: Optional[int],
        cached_state_index: Optional[int],
    ) -> None:
        kwargs = dict(path=self._state_path, config_hash=config_hash)
        if budget != self._max_budget:  # update the cache data
            new_state = [total_runtime, self._cumtime, budget, seed]
            _cache_state(new_state=new_state, update_index=cached_state_index, **kwargs)
        elif cached_state_index is not None:  # if None, newly start and train till the end, so no need to delete.
            _delete_state(index=cached_state_index, **kwargs)

    def _proc_output(self, eval_config: Dict[str, Any], budget: int, bench_data: Optional[Any]) -> Dict[str, float]:
        config_hash = hash(str(eval_config))
        kwargs = dict(config_hash=config_hash, budget=budget)
        cached_state, cached_state_index = self._get_cached_state_and_index(**kwargs)
        cached_runtime, _, _, seed = cached_state
        bench_data_kwargs = {} if bench_data is None else dict(bench_data=bench_data)
        output = self._func(eval_config=eval_config, budget=budget, seed=seed, **bench_data_kwargs)
        loss, total_runtime = output[self._loss_key], output[self._runtime_key]
        actual_runtime = max(0.0, total_runtime - cached_runtime)
        self._cumtime += actual_runtime
        self._update_state(total_runtime=total_runtime, cached_state_index=cached_state_index, seed=seed, **kwargs)
        return {self._loss_key: loss, self._runtime_key: actual_runtime}

    def __call__(self, eval_config: Dict[str, Any], budget: int, bench_data: Optional[Any] = None) -> Dict[str, float]:
        if self._terminated:
            return {self._loss_key: INF, self._runtime_key: INF}

        self._cumtime += time.time() - self._prev_timestamp  # sampling time
        output = self._proc_output(eval_config, budget, bench_data)
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=self._cumtime)
        _wait_until_next(path=self._cumtime_path, worker_id=self._worker_id)
        self._prev_timestamp = time.time()
        row = dict(loss=output[self._loss_key], cumtime=self._cumtime, index=self._index)
        _record_result(self._result_path, results=row)
        return output

    def finish(self) -> None:
        _record_cumtime(path=self._cumtime_path, worker_id=self._worker_id, cumtime=INF)
        self._terminated = True


class CentralWorker:
    def __init__(
        self,
        obj_func: _ObjectiveFunc,
        n_workers: int,
        max_budget: int,
        max_evals: int,
        subdir_name: str,
        loss_key: str = "loss",
        runtime_key: str = "runtime",
    ):
        worker_kwargs = dict(
            func=obj_func,
            n_workers=n_workers,
            subdir_name=subdir_name,
            max_budget=max_budget,
            loss_key=loss_key,
            runtime_key=runtime_key,
        )
        self._n_workers = n_workers
        self._workers: List[WorkerFunc]
        self._init_workers(worker_kwargs)

        self._max_evals = max_evals
        self._dir_name = self._workers[0].dir_name
        self._result_path = os.path.join(self._dir_name, RESULT_FILE_NAME)
        self._pid_to_index: Dict[int, int] = {}

    def _init_workers(self, worker_kwargs: Dict[str, Any]) -> None:
        pool = Pool()
        results = []
        for _ in range(self._n_workers):
            results.append(pool.apply_async(WorkerFunc, kwds=worker_kwargs))

        pool.close()
        pool.join()
        self._workers = [result.get() for result in results]

    def _init_alloc(self, pid: int) -> None:
        _path = os.path.join(self._dir_name, PROC_ALLOC_NAME)
        _allocate_proc_to_worker(path=_path, pid=pid)
        self._pid_to_index = _wait_proc_allocation(path=_path, n_workers=self._n_workers)

    def __call__(self, eval_config: Dict[str, Any], budget: int, bench_data: Optional[Any] = None) -> Dict:
        pid = os.getpid()
        if len(self._pid_to_index) != self._n_workers:
            self._init_alloc(pid)

        worker_index = self._pid_to_index[pid]
        output = self._workers[worker_index](eval_config=eval_config, budget=budget, bench_data=bench_data)
        if _is_simulator_terminated(self._result_path, max_evals=self._max_evals):
            self._workers[worker_index].finish()

        return output
