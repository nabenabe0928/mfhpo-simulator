from __future__ import annotations

import os
import time
from typing import Any

from benchmark_simulator._constants import (
    INF,
    _TIME_VALUES,
    _TimeNowDictType,
    _WorkerVars,
    _WrapperVars,
)
from benchmark_simulator._secure_proc import (
    _fetch_cumtimes,
    _fetch_timenow,
    _fetch_timestamps,
    _finish_worker_timer,
    _init_simulator,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
    _record_timenow,
    _record_timestamp,
    _start_timestamp,
    _start_worker_timer,
    _wait_all_workers,
    _wait_until_next,
)
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._utils import (
    _validate_fidel_args,
    _validate_fidels,
    _validate_fidels_continual,
    _validate_output,
)
from benchmark_simulator._trackers._config_tracker import _ConfigIDTracker
from benchmark_simulator._trackers._state_tracker import _StateTracker
from benchmark_simulator._utils import _generate_time_hash

import numpy as np


class _ObjectiveFuncWorker(_BaseWrapperInterface):
    """A worker class for each worker.
    This worker class is supposed to be instantiated for each worker.
    For example, if we use 4 workers for an optimization, then four instances should be created.

    Note:
        See benchmark_simulator/simulator.py to know variables shared across workers.
    """

    def __init__(self, wrapper_vars: _WrapperVars, worker_index: int | None, async_instantiations: bool):
        self._temp_worker_index = worker_index
        self._async_inst = async_instantiations
        super().__init__(wrapper_vars=wrapper_vars)

    def __repr__(self) -> str:
        return f"Worker-{self._worker_vars.worker_id}"

    def _init_worker(self, worker_id: str) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        self._state_tracker = _StateTracker(
            path=self._paths.state_cache, lock=self._lock, continual_max_fidel=self._wrapper_vars.continual_max_fidel
        )
        self._config_tracker = _ConfigIDTracker(path=self._paths.config_tracker, lock=self._lock)
        _init_simulator(dir_name=self.dir_name, worker_index=self._temp_worker_index)
        self._wait_till_init_simulator_finish()
        _start_worker_timer(path=self._paths.worker_cumtime, worker_id=worker_id, lock=self._lock)

    def _wait_till_init_simulator_finish(self) -> None:
        n_files = len(self._paths)
        n_repeats = 10000
        n_workers = self._wrapper_vars.n_workers
        while len(os.listdir(self.dir_name)) < n_files:  # pragma: no cover
            # In fact, below is covered by test, but not detected by pytest-cov
            time.sleep(1e-4)
            n_repeats -= 1
            if n_repeats == 0:
                msg = [
                    "The file initialization did not finish.",
                    "worker_index specified by users probably did not include 0.",
                    f"worker_index must be unique and in [0, {n_workers-1=}].",
                ]
                raise TimeoutError("\n".join(msg))

    def _alloc_index(self, worker_id: str) -> int:
        if not self._async_inst:
            assert self._temp_worker_index is not None  # mypy redefinition
            return self._temp_worker_index

        worker_id_to_index = _wait_all_workers(
            path=self._paths.worker_cumtime, n_workers=self._wrapper_vars.n_workers, lock=self._lock
        )
        time.sleep(1e-2)  # buffer before the optimization: DO NOT REMOVE
        return worker_id_to_index[worker_id]

    def _init_wrapper(self) -> None:
        continual_eval = self._wrapper_vars.continual_max_fidel is not None
        worker_id = _generate_time_hash()
        self._wrapper_vars.validate()
        _validate_fidel_args(continual_eval, fidel_keys=self._fidel_keys)
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

    def _validate(self) -> None:
        if self._crashed:
            raise InterruptedError(
                "The simulation is interrupted due to deadlock or the dead of at least one of the workers.\n"
                "This error could be avoided by increasing `max_waiting_time` (however, np.inf is discouraged).\n"
            )

    def _wait_other_workers(self) -> None:
        """
        Wait until the worker's cumulative runtime becomes the smallest.
        The smallest cumulative runtime implies that the order in the record will not disturbed
        even if the worker reports its results now.
        """
        _wait_until_next(
            path=self._paths.worker_cumtime,
            worker_id=self._worker_vars.worker_id,
            max_waiting_time=self._wrapper_vars.max_waiting_time,
            waiting_time=self._wrapper_vars.check_interval_time,
            lock=self._lock,
        )
        _record_timestamp(
            path=self._paths.timestamp,
            worker_id=self._worker_vars.worker_id,
            prev_timestamp=time.time(),
            lock=self._lock,
        )

    def _query_obj_func(
        self,
        eval_config: dict[str, Any],
        fidels: dict[str, int | float] | None,
        seed: int | None,
        prev_fidel: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        if self._wrapper_vars.store_config:
            self._used_config = eval_config.copy()
            self._used_config.update(
                **(fidels if fidels is not None else {}),
                **({"prev_fidel": prev_fidel} if prev_fidel is not None else {}),
                seed=seed,
            )

        return self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed, **data_to_scatter)

    def _proc_output_from_scratch(
        self, eval_config: dict[str, Any], fidels: dict[str, int | float] | None, **data_to_scatter: Any
    ) -> dict[str, float]:
        seed = self._worker_vars.rng.randint(1 << 30)
        results = self._query_obj_func(eval_config=eval_config, seed=seed, fidels=fidels, **data_to_scatter)
        _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
        self._cumtime += results[self.runtime_key]
        return {k: results[k] for k in self._worker_vars.stored_obj_keys}

    def _proc_output_from_existing_state(
        self, eval_config: dict[str, Any], fidel: int, config_id: int | None, **data_to_scatter: Any
    ) -> dict[str, float]:
        config_hash: int = hash(str(eval_config)) if config_id is None else int(config_id)
        cached_state, cached_state_index = self._state_tracker.get_cached_state_and_index(
            config_hash=config_hash, fidel=fidel, cumtime=self._cumtime, rng=self._worker_vars.rng
        )
        _fidels: dict[str, int | float] = {self._fidel_keys[0]: fidel}
        results = self._query_obj_func(
            eval_config=eval_config,
            seed=cached_state.seed,
            fidels=_fidels,
            prev_fidel=cached_state.fidel,
            **data_to_scatter,
        )
        _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
        total_runtime = results[self.runtime_key]
        actual_runtime = max(0.0, total_runtime - cached_state.runtime)
        self._cumtime += actual_runtime
        self._state_tracker.update_state(
            cumtime=self._cumtime,
            total_runtime=total_runtime,
            cached_state_index=cached_state_index,
            seed=cached_state.seed,
            config_hash=config_hash,
            fidel=fidel,
        )
        return {**{k: results[k] for k in self._obj_keys}, self.runtime_key: actual_runtime}

    def _proc_output(
        self,
        eval_config: dict[str, Any],
        fidels: dict[str, int | float] | None,
        config_id: int | None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        if not self._worker_vars.continual_eval:
            return self._proc_output_from_scratch(eval_config=eval_config, fidels=fidels, **data_to_scatter)

        # Otherwise, we try the continual evaluation
        fidel = _validate_fidels_continual(fidels)
        return self._proc_output_from_existing_state(
            eval_config=eval_config, fidel=fidel, config_id=config_id, **data_to_scatter
        )

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

    def _load_timestamps(self) -> float:
        timestamp_dict = _fetch_timestamps(self._paths.timestamp, lock=self._lock)
        worker_id = self._worker_vars.worker_id
        if worker_id not in timestamp_dict:  # Initialize the timestamp
            timestamp = time.time()
            _start_timestamp(
                path=self._paths.timestamp,
                worker_id=worker_id,
                prev_timestamp=timestamp,
                lock=self._lock,
            )
            return timestamp

        timenow_data = _fetch_timenow(path=self._paths.timenow, lock=self._lock)
        cumtime = _fetch_cumtimes(self._paths.worker_cumtime, lock=self._lock)[worker_id]
        # Consider the sampling time overlap
        self._cumtime = (
            max(cumtime, np.max(timenow_data["after_sample"][timenow_data["before_sample"] <= cumtime]))
            if not self._wrapper_vars.allow_parallel_sampling
            else cumtime
        )
        self._terminated = self._cumtime >= _TIME_VALUES.terminated - 1e-5
        self._crashed = self._cumtime >= _TIME_VALUES.crashed - 1e-5
        return timestamp_dict[worker_id]

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        prev_timestamp = self._load_timestamps()
        self._validate()
        _validate_fidels(
            fidels=fidels,
            fidel_keys=self._fidel_keys,
            use_fidel=self._worker_vars.use_fidel,
            continual_eval=self._worker_vars.continual_eval,
        )
        config_tracking = config_id is not None and self._wrapper_vars.config_tracking
        if config_tracking:  # validate the config_id to ensure the user implementation is correct
            assert config_id is not None  # mypy redefinition
            self._config_tracker.validate(config=eval_config, config_id=config_id)

        if self._terminated:
            return {**{k: INF for k in self._obj_keys}, self.runtime_key: INF}

        sampling_time = max(0.0, time.time() - prev_timestamp)
        timenow_data = _TimeNowDictType(before_sample=self._cumtime, after_sample=self._cumtime + sampling_time)
        _record_timenow(path=self._paths.timenow, timenow_data=timenow_data, lock=self._lock)

        self._cumtime += sampling_time
        results = self._proc_output(eval_config=eval_config, fidels=fidels, config_id=config_id, **data_to_scatter)
        self._post_proc(results)
        return results

    def _finish(self) -> None:
        """The method to close the worker instance.
        This method must be called before we finish the optimization.
        If not called, optimization modules are likely to hang.
        """
        _finish_worker_timer(path=self._paths.worker_cumtime, worker_id=self._worker_vars.worker_id, lock=self._lock)
        self._terminated = True
