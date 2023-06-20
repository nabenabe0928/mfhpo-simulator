from __future__ import annotations

import json
import os
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

from benchmark_simulator._constants import _StateType
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._utils import _validate_fidels_continual

import numpy as np


@dataclass(frozen=True)
class _ResultData:
    cumtime: float
    eval_config: dict[str, Any]
    results: dict[str, float]
    fidels: dict[str, int | float]
    seed: int | None


class AbstractAskTellOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        # Return: tuple[eval_config, fidels]
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
    ) -> None:
        raise NotImplementedError


class AskTellWorkerManager(_BaseWrapperInterface):
    def _init_wrapper(self) -> None:
        self._wrapper_vars.validate()
        self._n_workers = self._wrapper_vars.n_workers
        self._rng = np.random.RandomState(self._wrapper_vars.seed)
        self._cumtimes: np.ndarray = np.zeros(self._n_workers, dtype=np.float64)
        self._intermediate_states: dict[int, list[_StateType]] = {}
        self._pending_results: list[_ResultData | None] = [None] * self._n_workers
        self._seen_config_keys: list[str] = []

        self._results: dict[str, list[Any]] = {"worker_index": [], "cumtime": []}
        self._results.update({k: [] for k in self._obj_keys})
        if self._wrapper_vars.store_config:
            self._results.update({k: [] for k in self._fidel_keys + ["seed"]})

    def _fetch_prev_state_index(self, config_hash: int, fidel: int, worker_id: int) -> int | None:
        states = self._intermediate_states.get(config_hash, [])
        intermediate_avail = [state.cumtime <= self._cumtimes[worker_id] and state.fidel < fidel for state in states]
        return intermediate_avail.index(True) if any(intermediate_avail) else None

    def _fetch_prev_seed(self, config_hash: int, fidel: int, worker_id: int) -> int | None:
        prev_state_index = self._fetch_prev_state_index(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        if prev_state_index is None:
            return self._rng.randint(1 << 30)
        else:
            return self._intermediate_states[config_hash][prev_state_index].seed

    def _pop_old_state(self, config_hash: int, fidel: int, worker_id: int) -> _StateType | None:
        prev_state_index = self._fetch_prev_state_index(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        if prev_state_index is None:
            if config_hash not in self._intermediate_states:
                self._intermediate_states[config_hash] = []

            return None

        return self._intermediate_states[config_hash].pop(prev_state_index)

    def _proc(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
    ) -> tuple[dict[str, float], int | None]:
        continual_max_fidel = self._wrapper_vars.continual_max_fidel
        if continual_max_fidel is None:  # not continual learning
            seed = self._rng.randint(1 << 30)
            results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
            return results, seed

        fidel = _validate_fidels_continual(fidels)
        runtime_key = self._wrapper_vars.runtime_key
        config_hash: int = hash(str(eval_config))
        seed = self._fetch_prev_seed(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
        old_state = self._pop_old_state(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        old_state = _StateType(seed=seed) if old_state is None else old_state

        results[runtime_key] = max(0.0, results[runtime_key] - old_state.runtime)
        if fidel < continual_max_fidel:
            new_state = _StateType(
                runtime=results[runtime_key],
                cumtime=self._cumtimes[worker_id],
                fidel=fidel,
                seed=old_state.seed,
            )
            self._intermediate_states[config_hash].append(new_state)

        return results, seed

    def _proc_obj_func(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
    ) -> None:
        results, seed = self._proc(eval_config=eval_config, worker_id=worker_id, fidels=fidels)
        runtime_key = self._wrapper_vars.runtime_key
        self._cumtimes[worker_id] += results[runtime_key]
        self._pending_results[worker_id] = _ResultData(
            cumtime=self._cumtimes[worker_id],
            eval_config=eval_config,
            results=results,
            fidels=fidels if fidels is not None else {},
            seed=seed,
        )

    def _record_result_data(self, result_data: _ResultData, worker_id: int) -> None:
        prev_size = len(self._results["cumtime"])
        self._results["worker_index"].append(worker_id)
        self._results["cumtime"].append(result_data.cumtime)
        for k in self._obj_keys:
            self._results[k].append(result_data.results[k])

        if not self._wrapper_vars.store_config:
            return

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
    ) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        start = time.time()
        eval_config, fidels = opt.ask()
        sampling_time = time.time() - start
        self._cumtimes[worker_id] += sampling_time
        return eval_config, fidels

    def _tell_pending_result(self, opt: AbstractAskTellOptimizer, worker_id: int) -> None:
        result_data = self._pending_results[worker_id]
        if result_data is None:
            return

        self._record_result_data(result_data=result_data, worker_id=worker_id)
        opt.tell(eval_config=result_data.eval_config, results=result_data.results, fidels=result_data.fidels)
        self._pending_results[worker_id] = None

    def _save_results(self) -> None:
        os.makedirs(self.dir_name, exist_ok=True)
        with open(self._paths.result, mode="w") as f:
            json.dump(self._results, f)

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        """
        Start the simulation using only the main process.
        Unlike the other worker wrappers, each objective function will not run in parallel.
        Instead, we internally simulate the cumulative runtime for each worker.
        For this sake, the optimizer must take so-called ask-and-tell interface.
        It means that optimizer can communicate with this class via `ask` and `tell` methods.
        As long as the optimizer takes this interface, arbitrary optimizers can be used for this class.

        Although this class may not be able to guarantee the exact behavior using parallel optimization,
        this class is safer than the other wrappers because it is thread-safe.
        Furthermore, if users want to try a large n_workers, this class is much safer and executable.

        Args:
            opt (AbstractAskTellOptimizer):
                An optimizer that has `ask` and `tell` methods.
                For example, if we run a sequential optimization, the expected loop looks like:
                    for i in range(100):
                        eval_config, fidels = opt.ask()
                        results = obj_func(eval_config, fidels)
                        opt.tell(eval_config, results, fidels)
        """
        if not hasattr(opt, "ask") or not hasattr(opt, "tell"):
            raise ValueError(
                "opt must have `ask` and `tell` methods.\n"
                "Inherit `AbstractAskTellOptimizer` and encapsulate your optimizer instance in the child class."
            )

        worker_id = 0
        for _ in range(self._wrapper_vars.n_evals):
            eval_config, fidels = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id, fidels=fidels)
            worker_id = np.argmin(self._cumtimes)
            self._tell_pending_result(opt=opt, worker_id=worker_id)

        self._save_results()
