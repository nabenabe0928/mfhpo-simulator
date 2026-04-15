from __future__ import annotations

import time
from typing import Any
import warnings

import numpy as np

from src._constants import _ResultData
from src._constants import _WrapperVars
from src._constants import AbstractAskTellOptimizer
from src._constants import NEGLIGIBLE_SEC


class _AskTellWorkerManager:
    def __init__(self, wrapper_vars: _WrapperVars):
        self._wrapper_vars = wrapper_vars
        self._init_wrapper()

    def _init_wrapper(self) -> None:
        self._wrapper_vars.validate()

        self._start_time = time.time()
        self._timenow = 0.0
        self._cumtimes = np.zeros(self._wrapper_vars.n_workers, dtype=float)
        self._worker_indices = np.arange(self._wrapper_vars.n_workers)
        self._pending_results: list[_ResultData | None] = [None] * self._wrapper_vars.n_workers
        self._sampled_time: dict[str, list[float]] = {"before_sample": [], "after_sample": [], "worker_index": []}
        self._results: dict[str, list[Any]] = {
            "worker_index": [],
            "cumtime": [],
            "objectives": [],
            "actual_cumtime": [],
        }

    def _proc_obj_func(self, eval_config: dict[str, Any], worker_id: int) -> None:
        results = self._wrapper_vars.obj_func(eval_config=eval_config)
        self._cumtimes[worker_id] += results[-1]
        self._pending_results[worker_id] = _ResultData(
            cumtime=self._cumtimes[worker_id], eval_config=eval_config, results=results
        )

    def _record_result_data(self, result_data: _ResultData, worker_id: int) -> None:
        self._results["worker_index"].append(worker_id)
        self._results["cumtime"].append(result_data.cumtime)
        self._results["objectives"].append(result_data.results[:-1])
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
        if (
            not self._wrapper_vars.expensive_sampler
            and is_first_sample
            and self._cumtimes[worker_id] > NEGLIGIBLE_SEC
            and self._cumtimes[worker_id] != np.min(self._cumtimes[self._cumtimes > NEGLIGIBLE_SEC])
        ):
            raise TimeoutError(
                "The initialization of the optimizer must be cheaper than one objective evuation.\n"
                "In principle, n_workers is too large for the objective to simulate correctly.\n"
                "Please set expensive_sampler=True or a smaller n_workers, or use a cheaper initialization.\n"
            )
        return eval_config

    def _tell_pending_result(self, opt: AbstractAskTellOptimizer, worker_id: int) -> None:
        free_worker_idxs = np.array([worker_id], dtype=int)
        if self._wrapper_vars.expensive_sampler:
            before_eval = self._sampled_time["after_sample"][-1]
            free_worker_idxs = np.union1d(self._worker_indices[self._cumtimes <= before_eval], free_worker_idxs)
        else:
            warnings.warn(f"Use expensive_sampler=True for {self.__class__.__name__} as it is more precise")

        for _worker_id in free_worker_idxs.astype(int).tolist():
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
        if not hasattr(opt, "ask") or not hasattr(opt, "tell"):
            opt_cls = AbstractAskTellOptimizer
            error_lines = [
                "opt must have `ask` and `tell` methods.",
                f"Inherit `{opt_cls.__name__}` and encapsulate your optimizer instance in the child class.",
                "The description of `ask` method is as follows:",
                f"\033[32m{opt_cls.ask.__doc__}\033[0m",
                "The description of `tell` method is as follows:",
                f"\033[32m{opt_cls.tell.__doc__}\033[0m",
            ]
            raise ValueError("\n".join(error_lines))
        worker_id = 0
        for i in range(self._wrapper_vars.n_evals + self._wrapper_vars.n_workers - 1):
            eval_config = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id)
            worker_id = np.argmin(self._cumtimes).item()

            if i + 1 >= self._wrapper_vars.n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(opt=opt, worker_id=worker_id)

            if self._cumtimes[worker_id] > self._wrapper_vars.max_total_eval_time:  # exceed time limit
                break

        self._finalize_results()
