from __future__ import annotations

import time
from typing import Any
import warnings

import numpy as np

from src._constants import _ResultData
from src._constants import _StateType
from src._constants import _WorkerVars
from src._constants import _WrapperVars
from src._constants import AbstractAskTellOptimizer
from src._constants import NEGLIGIBLE_SEC
from src._state_tracker import _AskTellStateTracker
from src._validators import _raise_optimizer_init_error
from src._validators import _validate_fidel_args
from src._validators import _validate_fidels
from src._validators import _validate_fidels_continual
from src._validators import _validate_opt_class
from src._validators import _validate_output


def _two_dicts_almost_equal(d1: dict[str, Any], d2: dict[str, Any]) -> bool:
    """for atol and rtol, I referred to numpy.isclose"""
    if set(d1.keys()) != set(d2.keys()):
        return False

    for k in d1.keys():
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if not np.isclose(v1, v2):
                return False
        elif v1 != v2:
            return False

    return True


class _AskTellWorkerManager:
    def __init__(self, wrapper_vars: _WrapperVars):
        self._wrapper_vars = wrapper_vars
        self._obj_keys, self._runtime_key = wrapper_vars.obj_keys, wrapper_vars.runtime_key
        self._fidel_keys = [] if wrapper_vars.fidel_keys is None else wrapper_vars.fidel_keys[:]
        self._init_wrapper()

    @property
    def obj_keys(self) -> list[str]:
        return self._obj_keys[:]

    @property
    def runtime_key(self) -> str:
        return self._runtime_key

    @property
    def fidel_keys(self) -> list[str]:
        return self._fidel_keys[:]

    def _init_wrapper(self) -> None:
        self._worker_vars = _WorkerVars(
            continual_eval=self._wrapper_vars.continual_max_fidel is not None,
            use_fidel=self._wrapper_vars.fidel_keys is not None,
            rng=np.random.RandomState(self._wrapper_vars.seed),
            stored_obj_keys=list(set(self.obj_keys + [self.runtime_key])),
            worker_id="",
            worker_index=-1,
        )
        self._existing_configs: dict[str, dict[str, Any]] = {}
        self._state_tracker = _AskTellStateTracker(continual_max_fidel=self._wrapper_vars.continual_max_fidel)

        self._wrapper_vars.validate()
        _validate_fidel_args(continual_eval=self._worker_vars.continual_eval, fidel_keys=self._fidel_keys)

        self._start_time = time.time()
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
        if self._wrapper_vars.store_actual_cumtime:
            self._results.update({"actual_cumtime": []})

    def _validate_config_id(self, config: dict[str, Any], config_id: int) -> None:
        config_id_str = str(config_id)
        if config_id_str not in self._existing_configs:
            self._existing_configs[config_id_str] = config.copy()
            return

        existing_config = self._existing_configs[config_id_str]
        if not _two_dicts_almost_equal(existing_config, config):
            raise ValueError(
                f"{config_id=} already exists ({existing_config=}), but got the duplicated config_id for {config=}"
            )

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
        # NOTE(nabenabe0928): We update here to prevent the re-use of the previous state.
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

        if self._wrapper_vars.store_actual_cumtime:
            self._results["actual_cumtime"].append(time.time() - self._start_time)

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
            self._validate_config_id(config=eval_config, config_id=config_id)

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
            _raise_optimizer_init_error()

        return eval_config, fidels, config_id

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
            opt.tell(
                eval_config=result_data.eval_config,
                results=result_data.results,
                fidels=result_data.fidels,
                config_id=result_data.config_id,
            )
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
            eval_config, fidels, config_id = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id, fidels=fidels, config_id=config_id)
            worker_id = np.argmin(self._cumtimes)

            if i + 1 >= self._wrapper_vars.n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(opt=opt, worker_id=worker_id)

            if self._cumtimes[worker_id] > self._wrapper_vars.max_total_eval_time:  # exceed time limit
                break

        self._finalize_results()
