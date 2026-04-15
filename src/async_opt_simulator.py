from __future__ import annotations

import time
from typing import TYPE_CHECKING
import warnings

import numpy as np
import optuna


if TYPE_CHECKING:
    from typing import Final

    from optunahub.benchmarks import BaseProblem


NEGLIGIBLE_SEC: Final[float] = 1e-12


class AsyncOptBenchmarkSimulator:
    def __init__(self, n_workers: int, expensive_sampler: bool, allow_parallel_sampling: bool) -> None:
        """a simulator class for async optimization using zero-cost benchmark without waiting.

        Args:
            n_workers (int):
                The number of simulated workers. In other words, how many parallel workers to simulate.
            allow_parallel_sampling (bool):
                Whether sampling can happen in parallel.
            expensive_sampler (bool):
                Whether the optimizer is expensive relative to a function evaluation.
        """
        if allow_parallel_sampling and expensive_sampler:
            raise ValueError(
                "expensive_sampler and allow_parallel_sampling cannot be True simultaneously.\n"
                "Note that allow_parallel_sampling=True correctly handles expensive samplers"
                " if sampling happens in parallel."
            )
        self._n_workers = n_workers
        self._expensive_sampler = expensive_sampler
        self._allow_parallel_sampling = allow_parallel_sampling
        self._timenow = 0.0
        self._cumtimes = np.zeros(n_workers, dtype=float)
        self._worker_indices = np.arange(n_workers)
        self._pending_results: list[tuple[int, list[float]] | None] = [None] * n_workers
        self._after_sample_times: list[float] = []

    def _proc_obj_func(self, trial: optuna.Trial, problem: BaseProblem, worker_id: int) -> None:
        output = problem(trial)
        trial.set_user_attr("worker_id", worker_id)
        if "runtime" not in trial.user_attrs:
            raise KeyError(
                "`runtime` must be set from the problem side. Please override the objective to set the runtime."
            )

        self._cumtimes[worker_id] += float(trial.user_attrs["runtime"])
        trial.set_user_attr("cumtime", self._cumtimes[worker_id].item())
        self._pending_results[worker_id] = (trial.number, [output] if isinstance(output, float) else list(output))

    def _ask_with_timer(self, study: optuna.Study, problem: BaseProblem, worker_id: int) -> optuna.Trial:
        start = time.time()
        trial = study.ask(problem.search_space)
        sampling_time = time.time() - start
        is_first_sample = bool(self._cumtimes[worker_id] < NEGLIGIBLE_SEC)
        if self._allow_parallel_sampling:
            before_sample = self._cumtimes[worker_id]
            self._cumtimes[worker_id] = self._cumtimes[worker_id] + sampling_time
        else:
            before_sample = max(self._timenow, self._cumtimes[worker_id])
            self._timenow = before_sample + sampling_time
            self._cumtimes[worker_id] = self._timenow

        trial.set_user_attr("worker_index", worker_id)
        trial.set_user_attr("before_sample", before_sample)
        trial.set_user_attr("after_sample", self._cumtimes[worker_id])
        self._after_sample_times.append(self._cumtimes[worker_id])
        if (
            not self._expensive_sampler
            and is_first_sample
            and self._cumtimes[worker_id] > NEGLIGIBLE_SEC
            and self._cumtimes[worker_id] != np.min(self._cumtimes[self._cumtimes > NEGLIGIBLE_SEC])
        ):
            raise TimeoutError(
                "The initialization of the optimizer must be cheaper than one objective evuation.\n"
                "In principle, n_workers is too large for the objective to simulate correctly.\n"
                "Please set expensive_sampler=True or a smaller n_workers, or use a cheaper initialization.\n"
            )
        return trial

    def _tell_pending_result(self, study: optuna.Study, worker_id: int) -> None:
        free_worker_idxs = np.array([worker_id], dtype=int)
        if self._expensive_sampler:
            before_eval = self._after_sample_times[-1]
            free_worker_idxs = np.union1d(self._worker_indices[self._cumtimes <= before_eval], free_worker_idxs)
        else:
            warnings.warn(f"Use expensive_sampler=True for {self.__class__.__name__} as it is more precise")

        for _worker_id in free_worker_idxs.astype(int).tolist():
            result = self._pending_results[_worker_id]
            if result is None:
                continue

            trial_number, values = result
            study.tell(trial_number, values)
            self._pending_results[_worker_id] = None

    def optimize(
        self, study: optuna.Study, problem: BaseProblem, *, n_trials: int | None = None, timeout: float | None = None
    ) -> None:
        """
        Start the async optimization using zero-cost benchmark without any sleep.

        Args:
            n_trials (int):
                How many trials we would like to collect.
            timeout (float):
                The maximum total evaluation time for the optimization (in simulated time but not the actual runtime).
        """
        worker_id = 0
        n_trials = n_trials or 2**20  # Sufficiently large number to finish optimizing.
        timeout = timeout or float("inf")
        for i in range(n_trials + self._n_workers - 1):
            trial = self._ask_with_timer(study, problem, worker_id=worker_id)
            self._proc_obj_func(trial=trial, problem=problem, worker_id=worker_id)
            worker_id = np.argmin(self._cumtimes).item()
            if i + 1 >= self._n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(study=study, worker_id=worker_id)
            if self._cumtimes[worker_id] > timeout:  # exceed time limit
                break
