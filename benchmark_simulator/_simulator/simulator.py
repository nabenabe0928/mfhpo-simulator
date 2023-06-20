from __future__ import annotations

import os
import threading
import time
from multiprocessing import Pool
from typing import Any

from benchmark_simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._constants import INF, _StateType, _TIME_VALUES, _TimeStampDictType, _WorkerVars
from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _cache_state,
    _delete_state,
    _fetch_cache_states,
    _fetch_cumtimes,
    _fetch_proc_alloc,
    _fetch_timestamps,
    _finish_worker_timer,
    _init_simulator,
    _is_allocation_ready,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
    _record_timestamp,
    _start_timestamp,
    _start_worker_timer,
    _wait_all_workers,
    _wait_proc_allocation,
    _wait_until_next,
)
from benchmark_simulator._utils import _generate_time_hash

import numpy as np


class ObjectiveFuncWorker(_BaseWrapperInterface):
    """A worker class for each worker.
    This worker class is supposed to be instantiated for each worker.
    For example, if we use 4 workers for an optimization, then four instances should be created.
    """

    def __repr__(self) -> str:
        return f"Worker-{self._worker_vars.worker_id}"

    def _guarantee_no_hang(self) -> None:
        n_workers, n_evals = self._wrapper_vars.n_workers, self._wrapper_vars.n_evals
        n_actual_evals_in_opt = self._wrapper_vars.n_actual_evals_in_opt
        if n_actual_evals_in_opt < n_workers + n_evals:
            threshold = n_workers + n_evals
            # In fact, n_workers + n_evals - 1 is the real minimum threshold.
            raise ValueError(
                "Cannot guarantee that optimziers will not hang. "
                f"Use n_actual_evals_in_opt >= {threshold} (= n_evals + n_workers) at least. "
                "Note that our package cannot change your optimizer setting, so "
                "make sure that you changed your optimizer setting, but not only `n_actual_evals_in_opt`."
            )

    def _validate_fidel_args(self, continual_eval: bool) -> None:
        # Guarantee the sufficiency: continual_eval ==> len(fidel_keys) == 1
        if continual_eval and len(self._fidel_keys) != 1:
            raise ValueError(
                f"continual_max_fidel is valid only if fidel_keys has only one element, but got {self._fidel_keys}"
            )

    def _init_worker(self, worker_id: str) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        _init_simulator(dir_name=self.dir_name)
        _start_worker_timer(path=self._paths.worker_cumtime, worker_id=worker_id, lock=self._lock)

    def _alloc_index(self, worker_id: str) -> int:
        worker_id_to_index = _wait_all_workers(
            path=self._paths.worker_cumtime, n_workers=self._wrapper_vars.n_workers, lock=self._lock
        )
        time.sleep(1e-2)  # buffer before the optimization
        return worker_id_to_index[worker_id]

    def _init_wrapper(self) -> None:
        continual_eval = self._wrapper_vars.continual_max_fidel is not None
        worker_id = _generate_time_hash()
        self._guarantee_no_hang()
        self._validate_fidel_args(continual_eval)
        self._init_worker(worker_id)
        worker_index = self._alloc_index(worker_id)
        self._worker_vars = _WorkerVars(
            continual_eval=continual_eval,
            worker_id=worker_id,
            worker_index=worker_index,
            rng=np.random.RandomState(self._wrapper_vars.seed),
            use_fidel=self._wrapper_vars.fidel_keys is not None,
            stored_obj_keys=list(set(self.obj_keys + [self.runtime_key])),
        )

        # These variables change over time and must be either loaded from file system or updated.
        self._cumtime = 0.0
        self._terminated = False
        self._crashed = False
        self._used_config: dict[str, Any] = {}

    def _validate(self, fidels: dict[str, int | float] | None) -> None:
        if self._crashed:
            raise InterruptedError(
                "The simulation is interrupted due to deadlock or the dead of at least one of the workers.\n"
                "This error could be avoided by increasing `max_waiting_time` (however, np.inf is discouraged).\n"
            )
        if not self._worker_vars.use_fidel and fidels is not None:
            raise ValueError(
                "Objective function got keyword `fidels`, but fidel_keys was not provided in worker instantiation."
            )
        if self._worker_vars.use_fidel and fidels is None:
            raise ValueError(
                "Objective function did not get keyword `fidels`, but fidel_keys was provided in worker instantiation."
            )

    def _validate_output(self, results: dict[str, float]) -> None:
        keys_in_output = set(results.keys())
        keys = set(self._worker_vars.stored_obj_keys)
        if keys_in_output.intersection(keys) != keys:
            raise KeyError(
                f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
                f"but got {results}"
            )

    @staticmethod
    def _validate_provided_fidels(fidels: dict[str, int | float] | None) -> int:
        if fidels is None or len(fidels.values()) != 1:
            raise ValueError(
                f"fidels must have only one element when continual_max_fidel is provided, but got {fidels}"
            )

        fidel = next(iter(fidels.values()))
        if not isinstance(fidel, int):
            raise ValueError(f"Fidelity for continual evaluation must be integer, but got {fidel}")
        if fidel < 0:
            raise ValueError(f"Fidelity for continual evaluation must be non-negative, but got {fidel}")

        return fidel

    def _get_cached_state_and_index(self, config_hash: int, fidel: int) -> tuple[_StateType, int | None]:
        cached_states = _fetch_cache_states(path=self._paths.state_cache, config_hash=config_hash, lock=self._lock)
        intermediate_avail = [state.cumtime <= self._cumtime and state.fidel < fidel for state in cached_states]
        cached_state_index = intermediate_avail.index(True) if any(intermediate_avail) else None
        if cached_state_index is None:
            # initial seed, note: 1 << 30 is a huge number that fits 32bit.
            init_state = _StateType(seed=self._worker_vars.rng.randint(1 << 30))
            return init_state, None
        else:
            return cached_states[cached_state_index], cached_state_index

    def _update_state(
        self,
        config_hash: int,
        fidel: int,
        total_runtime: float,
        seed: int | None,
        cached_state_index: int | None,
    ) -> None:
        kwargs = dict(path=self._paths.state_cache, config_hash=config_hash, lock=self._lock)
        if fidel != self._wrapper_vars.continual_max_fidel:  # update the cache data
            new_state = _StateType(runtime=total_runtime, cumtime=self._cumtime, fidel=fidel, seed=seed)
            _cache_state(new_state=new_state, update_index=cached_state_index, **kwargs)  # type: ignore[arg-type]
        elif cached_state_index is not None:  # if None, newly start and train till the end, so no need to delete.
            _delete_state(index=cached_state_index, **kwargs)  # type: ignore[arg-type]

    def _wait_other_workers(self) -> None:
        """
        Wait until the worker's cumulative runtime becomes the smallest.
        The smallest cumulative runtime implies that the order in the record will not disturbed
        even if the worker reports its results now.
        """
        wait_start, worker_id = time.time(), self._worker_vars.worker_id
        max_waiting_time = self._wrapper_vars.max_waiting_time
        _wait_until_next(
            path=self._paths.worker_cumtime,
            worker_id=worker_id,
            max_waiting_time=max_waiting_time,
            waiting_time=self._wrapper_vars.check_interval_time,
            lock=self._lock,
        )
        _record_timestamp(
            path=self._paths.timestamp,
            worker_id=worker_id,
            prev_timestamp=time.time(),
            waited_time=time.time() - wait_start,
            lock=self._lock,
        )

    def _query_obj_func(
        self,
        eval_config: dict[str, Any],
        fidels: dict[str, int | float] | None,
        seed: int | None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        if self._wrapper_vars.store_config:
            self._used_config = eval_config.copy()
            self._used_config.update(**(fidels if fidels is not None else {}), seed=seed)

        return self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed, **data_to_scatter)

    def _proc_output_from_scratch(
        self, eval_config: dict[str, Any], fidels: dict[str, int | float] | None, **data_to_scatter: Any
    ) -> dict[str, float]:
        _fidels: dict[str, int | float] = {} if fidels is None else fidels.copy()
        if self._worker_vars.use_fidel and set(_fidels.keys()) != set(self._fidel_keys):
            raise KeyError(f"The keys in fidels must be identical to fidel_keys, but got {fidels}")

        seed = self._worker_vars.rng.randint(1 << 30)
        results = self._query_obj_func(eval_config=eval_config, seed=seed, fidels=fidels, **data_to_scatter)
        self._validate_output(results)
        self._cumtime += results[self.runtime_key]
        return {k: results[k] for k in self._worker_vars.stored_obj_keys}

    def _proc_output_from_existing_state(
        self, eval_config: dict[str, Any], fidel: int, **data_to_scatter: Any
    ) -> dict[str, float]:
        config_hash: int = hash(str(eval_config))
        cached_state, cached_state_index = self._get_cached_state_and_index(config_hash=config_hash, fidel=fidel)
        _fidels: dict[str, int | float] = {self._fidel_keys[0]: fidel}
        results = self._query_obj_func(
            eval_config=eval_config, seed=cached_state.seed, fidels=_fidels, **data_to_scatter
        )
        self._validate_output(results)
        total_runtime = results[self.runtime_key]
        actual_runtime = max(0.0, total_runtime - cached_state.runtime)
        self._cumtime += actual_runtime
        self._update_state(
            total_runtime=total_runtime,
            cached_state_index=cached_state_index,
            seed=cached_state.seed,
            config_hash=config_hash,
            fidel=fidel,
        )
        return {**{k: results[k] for k in self._obj_keys}, self.runtime_key: actual_runtime}

    def _proc_output(
        self, eval_config: dict[str, Any], fidels: dict[str, int | float] | None, **data_to_scatter: Any
    ) -> dict[str, float]:
        if not self._worker_vars.continual_eval:
            return self._proc_output_from_scratch(eval_config=eval_config, fidels=fidels, **data_to_scatter)

        # Otherwise, we try the continual evaluation
        fidel = self._validate_provided_fidels(fidels)
        return self._proc_output_from_existing_state(eval_config=eval_config, fidel=fidel, **data_to_scatter)

    def _post_proc(self, results: dict[str, float]) -> None:
        # First, record the simulated cumulative runtime after calling the objective
        _record_cumtime(
            path=self._paths.worker_cumtime,
            worker_id=self._worker_vars.worker_id,
            cumtime=self._cumtime,
            lock=self._lock,
        )
        # Wait till the cumulative runtime becomes the smallest
        self._wait_other_workers()

        row = dict(
            cumtime=self._cumtime,
            worker_index=self._worker_vars.worker_index,
            **{k: results[k] for k in self._obj_keys},
            **self._used_config,
        )
        # Record the results to the main database when the cumulative runtime is the smallest
        _record_result(
            self._paths.result, results=row, fixed=bool(not self._wrapper_vars.store_config), lock=self._lock
        )
        self._used_config = {}  # Make it empty
        if _is_simulator_terminated(self._paths.result, max_evals=self._wrapper_vars.n_evals, lock=self._lock):
            self._finish()

    def _load_timestamps(self) -> _TimeStampDictType:
        timestamp_dict = _fetch_timestamps(self._paths.timestamp, lock=self._lock)
        worker_id = self._worker_vars.worker_id
        if worker_id not in timestamp_dict:  # Initialize the timestamp
            init_timestamp = _TimeStampDictType(prev_timestamp=time.time(), waited_time=0.0)
            _start_timestamp(
                path=self._paths.timestamp,
                worker_id=worker_id,
                prev_timestamp=init_timestamp.prev_timestamp,
                lock=self._lock,
            )
            return init_timestamp

        timestamp = timestamp_dict[worker_id]
        self._cumtime = _fetch_cumtimes(self._paths.worker_cumtime, lock=self._lock)[worker_id]
        self._terminated = self._cumtime >= _TIME_VALUES.terminated - 1e-5
        self._crashed = self._cumtime >= _TIME_VALUES.crashed - 1e-5
        return timestamp

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang at the end.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (dict[str, int | float] | None):
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
            results (dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        timestamp = self._load_timestamps()
        self._validate(fidels=fidels)
        if self._terminated:
            return {**{k: INF for k in self._obj_keys}, self.runtime_key: INF}

        sampling_time = max(0.0, time.time() - timestamp.prev_timestamp - timestamp.waited_time)
        self._cumtime += sampling_time

        results = self._proc_output(eval_config, fidels, **data_to_scatter)
        self._post_proc(results)
        return results

    def _finish(self) -> None:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang.
        """
        _finish_worker_timer(path=self._paths.worker_cumtime, worker_id=self._worker_vars.worker_id, lock=self._lock)
        self._terminated = True


class CentralWorkerManager(_BaseWrapperInterface):
    """A central worker manager class.
    This class is supposed to be instantiated if the optimizer module uses multiprocessing.
    For example, Dask, multiprocessing, and joblib would need this class.
    This class recognizes each worker by process ID.
    Therefore, process ID for each worker must be always unique and identical.
    """

    def _init_wrapper(self) -> None:
        self._workers: list[ObjectiveFuncWorker]
        self._main_pid = os.getpid()
        self._init_workers()
        self._pid_to_index: dict[int, int] = {}

    def _init_workers(self) -> None:
        if os.path.exists(self.dir_name):
            raise FileExistsError(f"The directory `{self.dir_name}` already exists. Remove it first.")

        pool = Pool()
        results = []
        for _ in range(self._wrapper_vars.n_workers):
            results.append(pool.apply_async(ObjectiveFuncWorker, kwds=self._wrapper_vars.__dict__))

        pool.close()
        pool.join()
        self._workers = [result.get() for result in results]

    def _init_alloc(self, pid: int) -> None:
        path = self._paths.proc_alloc
        if not _is_allocation_ready(path=path, n_workers=self._wrapper_vars.n_workers, lock=self._lock):
            _allocate_proc_to_worker(path=path, pid=pid, lock=self._lock)
            self._pid_to_index = _wait_proc_allocation(
                path=path, n_workers=self._wrapper_vars.n_workers, lock=self._lock
            )
        else:
            self._pid_to_index = _fetch_proc_alloc(path=path, lock=self._lock)

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        """The meta-wrapper method of the objective function method in WorkerFunc instances.

        This method recognizes each WorkerFunc by process ID and call the corresponding worker based on the ID.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (dict[str, int | float] | None):
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
            results (dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        pid = os.getpid()
        pid = threading.get_ident() if pid == self._main_pid else pid
        if len(self._pid_to_index) != self._wrapper_vars.n_workers:
            self._init_alloc(pid)

        if pid not in self._pid_to_index:
            raise ProcessLookupError(
                f"An unknown process/thread with ID {pid} was specified.\n"
                "It is likely that one of the workers crashed and a new worker was added.\n"
                f"However, worker additions are not allowed in {self.__class__.__name__}."
            )

        worker_index = self._pid_to_index[pid]
        results = self._workers[worker_index](eval_config=eval_config, fidels=fidels, **data_to_scatter)
        return results
