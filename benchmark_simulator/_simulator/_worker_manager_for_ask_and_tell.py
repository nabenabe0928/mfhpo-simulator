from __future__ import annotations

import os
import time
from typing import Any

from benchmark_simulator._constants import (
    AbstractAskTellOptimizer,
    NEGLIGIBLE_SEC,
    _ResultData,
    _StateType,
    _WorkerVars,
)
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._utils import (
    _raise_optimizer_init_error,
    _validate_fidel_args,
    _validate_fidels,
    _validate_fidels_continual,
    _validate_opt_class,
    _validate_output,
)
from benchmark_simulator._trackers._config_tracker import _AskTellConfigIDTracker
from benchmark_simulator._trackers._state_tracker import _AskTellStateTracker

import numpy as np

import ujson as json  # type: ignore


class _AskTellWorkerManager(_BaseWrapperInterface):
    def _init_wrapper(self) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        self._worker_vars = _WorkerVars(
            continual_eval=self._wrapper_vars.continual_max_fidel is not None,
            use_fidel=self._wrapper_vars.fidel_keys is not None,
            rng=np.random.RandomState(self._wrapper_vars.seed),
            stored_obj_keys=list(set(self.obj_keys + [self.runtime_key])),
            worker_id="",
            worker_index=-1,
        )
        self._config_tracker = _AskTellConfigIDTracker()
        self._state_tracker = _AskTellStateTracker(continual_max_fidel=self._wrapper_vars.continual_max_fidel)

        self._wrapper_vars.validate()
        _validate_fidel_args(continual_eval=self._worker_vars.continual_eval, fidel_keys=self._fidel_keys)

        self._timenow = 0.0
        self._cumtimes: np.ndarray = np.zeros(self._wrapper_vars.n_workers, dtype=np.float64)
        self._worker_indices = np.arange(self._wrapper_vars.n_workers)
        self._pending_results: list[_ResultData | None] = [None] * self._wrapper_vars.n_workers
        self._seen_config_keys: list[str] = []
        self._sampled_time: dict[str, list[float]] = {"before_sample": [], "after_sample": [], "worker_index": []}
        self._results: dict[str, list[Any]] = {"worker_index": [], "cumtime": []}
        self._results.update({k: [] for k in self._obj_keys})
        if self._wrapper_vars.store_config:
            self._results.update({k: [] for k in self._fidel_keys + ["seed"]})
            if self._wrapper_vars.continual_max_fidel is not None:
                self._results["prev_fidel"] = []

    def _proc(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
        config_id: int | None,
    ) -> tuple[dict[str, float], int | None, int | None]:
        if not self._worker_vars.continual_eval:  # not continual learning
            seed = self._worker_vars.rng.randint(1 << 30)
            results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
            _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
            return results, seed, None

        fidel = _validate_fidels_continual(fidels)
        runtime_key = self._wrapper_vars.runtime_key
        config_hash = int(hash(str(eval_config))) if config_id is None else config_id
        seed = self._state_tracker.fetch_prev_seed(
            config_hash=config_hash,
            fidel=fidel,
            cumtime=self._cumtimes[worker_id],
            rng=self._worker_vars.rng,
        )
        results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
        _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
        old_state = self._state_tracker.pop_old_state(
            config_hash=config_hash, fidel=fidel, cumtime=self._cumtimes[worker_id]
        )
        old_state = _StateType(seed=seed) if old_state is None else old_state

        results[runtime_key] = max(0.0, results[runtime_key] - old_state.runtime)
        self._state_tracker.update_state(
            config_hash=config_hash,
            fidel=fidel,
            runtime=results[runtime_key],
            cumtime=self._cumtimes[worker_id],
            seed=old_state.seed,
        )

        return results, seed, old_state.fidel

    def _proc_obj_func(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
        config_id: int | None,
    ) -> None:
        _validate_fidels(
            fidels=fidels,
            fidel_keys=self._fidel_keys,
            use_fidel=self._worker_vars.use_fidel,
            continual_eval=self._worker_vars.continual_eval,
        )
        results, seed, prev_fidel = self._proc(
            eval_config=eval_config, worker_id=worker_id, fidels=fidels, config_id=config_id
        )
        runtime_key = self._wrapper_vars.runtime_key
        self._cumtimes[worker_id] += results[runtime_key]
        self._pending_results[worker_id] = _ResultData(
            cumtime=self._cumtimes[worker_id],
            eval_config=eval_config,
            results=results,
            fidels=fidels if fidels is not None else {},
            seed=seed,
            config_id=config_id,
            prev_fidel=prev_fidel,
        )

    def _record_result_data(self, result_data: _ResultData, worker_id: int) -> None:
        prev_size = len(self._results["cumtime"])
        self._results["worker_index"].append(worker_id)
        self._results["cumtime"].append(result_data.cumtime)
        for k in self._obj_keys:
            self._results[k].append(result_data.results[k])

        if not self._wrapper_vars.store_config:
            return
        if result_data.prev_fidel is not None:
            self._results["prev_fidel"].append(result_data.prev_fidel)

        self._results["seed"].append(result_data.seed)
        for k in self._fidel_keys:
            self._results[k].append(result_data.fidels[k])

        eval_config = result_data.eval_config
        unseen_keys = [k for k in eval_config.keys() if k not in self._seen_config_keys]
        self._seen_config_keys.extend(unseen_keys)
        for k in self._seen_config_keys:
            val = eval_config.get(k, None)
            if k in unseen_keys:
                self._results[k] = [None] * prev_size + [val]
            else:
                self._results[k].append(val)

    def _ask_with_timer(
        self,
        opt: AbstractAskTellOptimizer,
        worker_id: int,
    ) -> tuple[dict[str, Any], dict[str, int | float] | None, int | None]:
        start = time.time()
        eval_config, fidels, config_id = opt.ask()
        sampling_time = time.time() - start
        config_tracking = config_id is not None and self._wrapper_vars.config_tracking
        if config_tracking:  # validate the config_id to ensure the user implementation is correct
            assert config_id is not None  # mypy redefinition
            self._config_tracker.validate(config=eval_config, config_id=config_id)

        is_first_sample = bool(self._cumtimes[worker_id] < NEGLIGIBLE_SEC)
        if self._wrapper_vars.allow_parallel_sampling:
            self._cumtimes[worker_id] = self._cumtimes[worker_id] + sampling_time
        else:
            self._timenow = max(self._timenow, self._cumtimes[worker_id]) + sampling_time
            self._cumtimes[worker_id] = self._timenow

        self._sampled_time["worker_index"].append(worker_id)
        self._sampled_time["before_sample"].append(self._cumtimes[worker_id] - sampling_time)
        self._sampled_time["after_sample"].append(self._cumtimes[worker_id])

        if (
            not self._wrapper_vars.expensive_sampler
            and is_first_sample
            and self._cumtimes[worker_id] != np.min(self._cumtimes[self._cumtimes > NEGLIGIBLE_SEC])
        ):
            _raise_optimizer_init_error()

        return eval_config, fidels, config_id

    def _tell_pending_result(self, opt: AbstractAskTellOptimizer, worker_id: int) -> None:
        free_worker_idxs = np.array([worker_id], dtype=np.int32)
        if self._wrapper_vars.expensive_sampler:
            before_eval = self._sampled_time["after_sample"][-1]
            free_worker_idxs = np.union1d(self._worker_indices[self._cumtimes <= before_eval], free_worker_idxs)

        for _worker_id in free_worker_idxs.astype(np.int32):
            result_data = self._pending_results[_worker_id]
            if result_data is None:
                continue

            self._record_result_data(result_data=result_data, worker_id=_worker_id)
            opt.tell(
                eval_config=result_data.eval_config,
                results=result_data.results,
                fidels=result_data.fidels,
                config_id=result_data.config_id,
            )
            self._pending_results[_worker_id] = None

    def _save_results(self) -> None:
        cumtime = np.array(self._results["cumtime"])
        order = np.argsort(cumtime) if self._wrapper_vars.expensive_sampler else np.arange(cumtime.size)
        with open(self._paths.result, mode="w") as f:
            json.dump({k: np.asarray(v)[order].tolist() for k, v in self._results.items()}, f, indent=4)

        with open(self._paths.sampled_time, mode="w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self._sampled_time.items()}, f, indent=4)

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        _validate_opt_class(opt)
        worker_id = 0
        for i in range(self._wrapper_vars.n_evals + self._wrapper_vars.n_workers - 1):
            eval_config, fidels, config_id = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id, fidels=fidels, config_id=config_id)
            worker_id = np.argmin(self._cumtimes)

            if i + 1 >= self._wrapper_vars.n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(opt=opt, worker_id=worker_id)

            if self._cumtimes[worker_id] > self._wrapper_vars.max_total_eval_time:  # exceed time limit
                break

        self._save_results()
