from __future__ import annotations

import time
from typing import Any
import warnings

import numpy as np

from src._constants import _ResultData
from src._constants import _WrapperVars
from src._constants import AbstractAskTellOptimizer
from src._constants import NEGLIGIBLE_SEC
from src._validators import _validate_opt_class
from src._validators import _validate_output


class _AskTellWorkerManager:
    def __init__(self, wrapper_vars: _WrapperVars):
        self._wrapper_vars = wrapper_vars
        self._obj_keys, self._runtime_key = wrapper_vars.obj_keys, wrapper_vars.runtime_key
        self._init_wrapper()

    @property
    def obj_keys(self) -> list[str]:
        return self._obj_keys[:]

    @property
    def runtime_key(self) -> str:
        return self._runtime_key

    def _init_wrapper(self) -> None:
        self._stored_obj_keys = list(set(self.obj_keys + [self.runtime_key]))

        self._wrapper_vars.validate()

        self._start_time = time.time()
        self._timenow = 0.0
        self._cumtimes: np.ndarray = np.zeros(self._wrapper_vars.n_workers, dtype=np.float64)
        self._worker_indices = np.arange(self._wrapper_vars.n_workers)
        self._pending_results: list[_ResultData | None] = [None] * self._wrapper_vars.n_workers
        self._sampled_time: dict[str, list[float]] = {"before_sample": [], "after_sample": [], "worker_index": []}
        self._results: dict[str, list[Any]] = {"worker_index": [], "cumtime": []}
        self._results.update({k: [] for k in self._obj_keys})
        if self._wrapper_vars.store_actual_cumtime:
            self._results.update({"actual_cumtime": []})

    def _proc(self, eval_config: dict[str, Any]) -> dict[str, float]:
        results = self._wrapper_vars.obj_func(eval_config=eval_config)
        _validate_output(results, stored_obj_keys=self._stored_obj_keys)
        return results

    def _proc_obj_func(self, eval_config: dict[str, Any], worker_id: int) -> None:
        results = self._proc(eval_config=eval_config)
        runtime_key = self._wrapper_vars.runtime_key
        self._cumtimes[worker_id] += results[runtime_key]
        self._pending_results[worker_id] = _ResultData(
            cumtime=self._cumtimes[worker_id], eval_config=eval_config, results=results
        )

    def _record_result_data(self, result_data: _ResultData, worker_id: int) -> None:
        self._results["worker_index"].append(worker_id)
        self._results["cumtime"].append(result_data.cumtime)
        for k in self._obj_keys:
            self._results[k].append(result_data.results[k])

        if self._wrapper_vars.store_actual_cumtime:
            self._results["actual_cumtime"].append(time.time() - self._start_time)

    def _ask_with_timer(self, opt: AbstractAskTellOptimizer, worker_id: int) -> dict[str, Any]:
        start = time.time()
        eval_config = opt.ask()
        sampling_time = time.time() - start

        is_first_sample = bool(self._cumtimes[worker_id] < NEGLIGIBLE_SEC)
        if self._wrapper_vars.allow_parallel_sampling:
            before_sample = self._cumtimes[worker_id]
            self._cumtimes[worker_id] = self._cumtimes[worker_id] + sampling_time
        else:
            before_sample = max(self._timenow, self._cumtimes[worker_id])
            self._timenow = before_sample + sampling_time
            self._cumtimes[worker_id] = self._timenow

        self._sampled_time["worker_index"].append(worker_id)
        self._sampled_time["before_sample"].append(before_sample)
        self._sampled_time["after_sample"].append(self._cumtimes[worker_id])

        positive_cumtimes = self._cumtimes[self._cumtimes > NEGLIGIBLE_SEC]
        if (
            not self._wrapper_vars.expensive_sampler
            and is_first_sample
            and positive_cumtimes.size > 0
            and self._cumtimes[worker_id] > NEGLIGIBLE_SEC
            and self._cumtimes[worker_id] != np.min(positive_cumtimes)
        ):
            raise TimeoutError(
                "The initialization of the optimizer must be cheaper than one objective evuation.\n"
                "In principle, n_workers is too large for the objective to simulate correctly.\n"
                "Please set expensive_sampler=True or a smaller n_workers, or use a cheaper initialization.\n"
            )

        return eval_config

    def _tell_pending_result(self, opt: AbstractAskTellOptimizer, worker_id: int) -> None:
        free_worker_idxs = np.array([worker_id], dtype=np.int32)
        if self._wrapper_vars.expensive_sampler:
            before_eval = self._sampled_time["after_sample"][-1]
            free_worker_idxs = np.union1d(self._worker_indices[self._cumtimes <= before_eval], free_worker_idxs)
        else:
            warnings.warn(f"Use expensive_sampler=True for {self.__class__.__name__} as it is more precise")

        for _worker_id in free_worker_idxs.astype(np.int32):
            result_data = self._pending_results[_worker_id]
            if result_data is None:
                continue

            self._record_result_data(result_data=result_data, worker_id=_worker_id)
            opt.tell(eval_config=result_data.eval_config, results=result_data.results)
            self._pending_results[_worker_id] = None

    def _finalize_results(self) -> None:
        cumtime = np.array(self._results["cumtime"])
        order = np.argsort(cumtime) if self._wrapper_vars.expensive_sampler else np.arange(cumtime.size)
        self._final_results = {k: np.asarray(v)[order].tolist() for k, v in self._results.items()}
        self._final_sampled_time = {k: np.asarray(v).tolist() for k, v in self._sampled_time.items()}

    def get_results(self) -> dict[str, list[int | float | str | bool]]:
        return self._final_results

    def get_optimizer_overhead(self) -> dict[str, list[float]]:
        return self._final_sampled_time

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        _validate_opt_class(opt)
        worker_id = 0
        for i in range(self._wrapper_vars.n_evals + self._wrapper_vars.n_workers - 1):
            eval_config = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id)
            worker_id = np.argmin(self._cumtimes)

            if i + 1 >= self._wrapper_vars.n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(opt=opt, worker_id=worker_id)

            if self._cumtimes[worker_id] > self._wrapper_vars.max_total_eval_time:  # exceed time limit
                break

        self._finalize_results()
