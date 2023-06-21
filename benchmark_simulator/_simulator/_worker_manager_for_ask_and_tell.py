from __future__ import annotations

import json
import os
import time
from typing import Any

from benchmark_simulator._constants import AbstractAskTellOptimizer, _ResultData, _StateType, _WorkerVars
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._utils import (
    _validate_fidel_args,
    _validate_fidels,
    _validate_fidels_continual,
    _validate_output,
)

import numpy as np


class AskTellWorkerManager(_BaseWrapperInterface):
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

        self._wrapper_vars.validate()
        _validate_fidel_args(continual_eval=self._worker_vars.continual_eval, fidel_keys=self._fidel_keys)

        self._timenow = 0.0
        self._cumtimes: np.ndarray = np.zeros(self._wrapper_vars.n_workers, dtype=np.float64)
        self._intermediate_states: dict[int, list[_StateType]] = {}
        self._pending_results: list[_ResultData | None] = [None] * self._wrapper_vars.n_workers
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
            return self._worker_vars.rng.randint(1 << 30)
        else:
            return self._intermediate_states[config_hash][prev_state_index].seed

    def _pop_old_state(self, config_hash: int, fidel: int, worker_id: int) -> _StateType | None:
        prev_state_index = self._fetch_prev_state_index(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        if prev_state_index is None:
            return None

        old_state = self._intermediate_states[config_hash].pop(prev_state_index)
        if len(self._intermediate_states[config_hash]) == 0:
            # Remove the empty set
            self._intermediate_states.pop(config_hash)

        return old_state

    def _proc(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
    ) -> tuple[dict[str, float], int | None]:
        continual_max_fidel = self._wrapper_vars.continual_max_fidel
        if not self._worker_vars.continual_eval:  # not continual learning
            seed = self._worker_vars.rng.randint(1 << 30)
            results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
            _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
            return results, seed

        fidel = _validate_fidels_continual(fidels)
        runtime_key = self._wrapper_vars.runtime_key
        config_hash = int(hash(str(eval_config)))
        seed = self._fetch_prev_seed(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        results = self._wrapper_vars.obj_func(eval_config=eval_config, fidels=fidels, seed=seed)
        _validate_output(results, stored_obj_keys=self._worker_vars.stored_obj_keys)
        old_state = self._pop_old_state(config_hash=config_hash, fidel=fidel, worker_id=worker_id)
        old_state = _StateType(seed=seed) if old_state is None else old_state

        results[runtime_key] = max(0.0, results[runtime_key] - old_state.runtime)
        assert continual_max_fidel is not None  # mypy redefinition
        if fidel < continual_max_fidel:
            new_state = _StateType(
                runtime=results[runtime_key],
                cumtime=self._cumtimes[worker_id],
                fidel=fidel,
                seed=old_state.seed,
            )
            if config_hash in self._intermediate_states:
                self._intermediate_states[config_hash].append(new_state)
            else:
                self._intermediate_states[config_hash] = [new_state]

        return results, seed

    def _proc_obj_func(
        self,
        eval_config: dict[str, Any],
        worker_id: int,
        fidels: dict[str, int | float] | None,
    ) -> None:
        _validate_fidels(
            fidels=fidels,
            fidel_keys=self._fidel_keys,
            use_fidel=self._worker_vars.use_fidel,
            continual_eval=self._worker_vars.continual_eval,
        )
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
        self._timenow = max(self._timenow, self._cumtimes[worker_id]) + sampling_time
        self._cumtimes[worker_id] = self._timenow
        return eval_config, fidels

    def _tell_pending_result(self, opt: AbstractAskTellOptimizer, worker_id: int) -> None:
        result_data = self._pending_results[worker_id]
        if result_data is None:
            return

        self._record_result_data(result_data=result_data, worker_id=worker_id)
        opt.tell(eval_config=result_data.eval_config, results=result_data.results, fidels=result_data.fidels)
        self._pending_results[worker_id] = None

    def _save_results(self) -> None:
        with open(self._paths.result, mode="w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self._results.items()}, f, indent=4)

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
            example_url = "https://github.com/nabenabe0928/mfhpo-simulator/blob/main/examples/"
            raise ValueError(
                "opt must have `ask` and `tell` methods.\n"
                f"Inherit `{AbstractAskTellOptimizer.__name__}` and \n"
                "encapsulate your optimizer instance in the child class.\n"
                "The description of `ask` method is as follows:\n"
                f"\033[32m{AbstractAskTellOptimizer.ask.__doc__}\033[0m\n"
                "The description of `tell` method is as follows:\n"
                f"\033[32m{AbstractAskTellOptimizer.tell.__doc__}\033[0m\n"
                f"See {example_url} for more details."
            )

        worker_id = 0
        for _ in range(self._wrapper_vars.n_evals + self._wrapper_vars.n_workers - 1):
            eval_config, fidels = self._ask_with_timer(opt=opt, worker_id=worker_id)
            self._proc_obj_func(eval_config=eval_config, worker_id=worker_id, fidels=fidels)
            worker_id = np.argmin(self._cumtimes)
            self._tell_pending_result(opt=opt, worker_id=worker_id)

        self._save_results()
