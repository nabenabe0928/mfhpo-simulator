from __future__ import annotations

from typing import Any

from benchmark_simulator._constants import AbstractAskTellOptimizer, _StateType
from benchmark_simulator._secure_proc import (
    _cache_state,
    _delete_state,
    _fetch_cache_states,
    _fetch_existing_configs,
    _record_existing_configs,
)
from benchmark_simulator._utils import _SecureLock

import numpy as np


def _two_dicts_almost_equal(d1: dict[str, Any], d2: dict[str, Any]) -> bool:
    """for atol and rtol, I referred to numpy.isclose"""
    if set(d1.keys()) == set(d2.keys()):
        return False

    for k in d1.keys():
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if not np.isclose(v1, v2):
                return False
        elif v1 != v2:
            return False

    return True


def _validate_continual_max_fidel(continual_max_fidel: int | None, cls_name: str) -> None:
    if not isinstance(continual_max_fidel, int):
        raise ValueError(f"continual_max_fidel must be int for {cls_name}")


class _ConfigIDTracker:
    def __init__(self, path: str, lock: _SecureLock):
        self._path = path
        self._lock = lock

    def _fetch_existing_configs(self) -> dict[str, dict[str, Any]]:
        return _fetch_existing_configs(path=self._path, lock=self._lock)

    def _record_existing_configs(self, config_id_str: str, config: dict[str, Any]) -> None:
        _record_existing_configs(path=self._path, config_id_str=config_id_str, config=config, lock=self._lock)

    def validate(self, config: dict[str, Any], config_id: int) -> None:
        config_id_str = str(config_id)
        existing_configs = self._fetch_existing_configs()
        if config_id_str not in existing_configs:
            self._record_existing_configs(config_id_str=config_id_str, config=config)
            return

        existing_config = existing_configs[config_id_str]
        if not _two_dicts_almost_equal(existing_config, config):
            raise ValueError(
                f"{config_id=} already exists ({existing_config=}), but got the duplicated config_id for {config=}"
            )


class _AskTellConfigIDTracker(_ConfigIDTracker):
    def __init__(self) -> None:
        self._existing_configs: dict[str, dict[str, Any]] = {}

    def _fetch_existing_configs(self) -> dict[str, dict[str, Any]]:
        return self._existing_configs

    def _record_existing_configs(self, config_id_str: str, config: dict[str, Any]) -> None:
        self._existing_configs[config_id_str] = config.copy()


class _AskTellStateTracker:
    def __init__(self, continual_max_fidel: int | None):
        self._intermediate_states: dict[int, list[_StateType]] = {}
        self._continual_max_fidel = continual_max_fidel

    def _fetch_prev_state_index(self, config_hash: int, fidel: int, cumtime: float) -> int | None:
        states = self._intermediate_states.get(config_hash, [])
        # This guarantees that `cached_state_index` yields the max fidel available in the cache
        intermediate_avail = [state.cumtime <= cumtime and state.fidel < fidel for state in states]
        return intermediate_avail.index(True) if any(intermediate_avail) else None

    def fetch_prev_seed(self, config_hash: int, fidel: int, cumtime: float, rng: np.random.RandomState) -> int | None:
        prev_state_index = self._fetch_prev_state_index(config_hash=config_hash, fidel=fidel, cumtime=cumtime)
        if prev_state_index is None:
            return rng.randint(1 << 30)
        else:
            return self._intermediate_states[config_hash][prev_state_index].seed

    def pop_old_state(self, config_hash: int, fidel: int, cumtime: float) -> _StateType | None:
        prev_state_index = self._fetch_prev_state_index(config_hash=config_hash, fidel=fidel, cumtime=cumtime)
        if prev_state_index is None:
            return None

        old_state = self._intermediate_states[config_hash].pop(prev_state_index)
        if len(self._intermediate_states[config_hash]) == 0:
            # Remove the empty set
            self._intermediate_states.pop(config_hash)

        return old_state

    def update_state(
        self,
        config_hash: int,
        fidel: int,
        runtime: float,
        cumtime: float,
        seed: int | None,
    ) -> None:
        _validate_continual_max_fidel(continual_max_fidel=self._continual_max_fidel, cls_name=self.__class__.__name__)
        assert isinstance(self._continual_max_fidel, int)  # mypy redefinition
        if fidel < self._continual_max_fidel:
            new_state = _StateType(runtime=runtime, cumtime=cumtime, fidel=fidel, seed=seed)
            if config_hash in self._intermediate_states:
                self._intermediate_states[config_hash].append(new_state)
            else:
                self._intermediate_states[config_hash] = [new_state]


class _StateTracker:
    def __init__(self, path: str, lock: _SecureLock, continual_max_fidel: int | None):
        self._path = path
        self._lock = lock
        self._continual_max_fidel = continual_max_fidel

    def get_cached_state_and_index(
        self, config_hash: int, fidel: int, cumtime: float, rng: np.random.RandomState
    ) -> tuple[_StateType, int | None]:
        _validate_continual_max_fidel(continual_max_fidel=self._continual_max_fidel, cls_name=self.__class__.__name__)
        assert isinstance(self._continual_max_fidel, int)  # mypy redefinition
        cached_states = _fetch_cache_states(path=self._path, config_hash=config_hash, lock=self._lock)
        intermediate_avail = [state.cumtime <= cumtime and state.fidel < fidel for state in cached_states]
        # This guarantees that `cached_state_index` yields the max fidel available in the cache
        cached_state_index = intermediate_avail.index(True) if any(intermediate_avail) else None
        if cached_state_index is None:
            # initial seed, note: 1 << 30 is a huge number that fits 32bit.
            init_state = _StateType(seed=rng.randint(1 << 30))
            return init_state, None
        else:
            return cached_states[cached_state_index], cached_state_index

    def update_state(
        self,
        cumtime: float,
        config_hash: int,
        fidel: int,
        total_runtime: float,
        seed: int | None,
        cached_state_index: int | None,
    ) -> None:
        _validate_continual_max_fidel(continual_max_fidel=self._continual_max_fidel, cls_name=self.__class__.__name__)
        kwargs = dict(path=self._path, config_hash=config_hash, lock=self._lock)
        if fidel != self._continual_max_fidel:  # update the cache data
            new_state = _StateType(runtime=total_runtime, cumtime=cumtime, fidel=fidel, seed=seed)
            _cache_state(new_state=new_state, update_index=cached_state_index, **kwargs)  # type: ignore[arg-type]
        elif cached_state_index is not None:  # if None, newly start and train till the end, so no need to delete.
            _delete_state(index=cached_state_index, **kwargs)  # type: ignore[arg-type]


def _validate_opt_class(opt: AbstractAskTellOptimizer) -> None:
    if not hasattr(opt, "ask") or not hasattr(opt, "tell"):
        example_url = "https://github.com/nabenabe0928/mfhpo-simulator/blob/main/examples/ask_and_tell/"
        opt_cls = AbstractAskTellOptimizer
        error_lines = [
            "opt must have `ask` and `tell` methods.",
            f"Inherit `{opt_cls.__name__}` and encapsulate your optimizer instance in the child class.",
            "The description of `ask` method is as follows:",
            f"\033[32m{opt_cls.ask.__doc__}\033[0m",
            "The description of `tell` method is as follows:",
            f"\033[32m{opt_cls.tell.__doc__}\033[0m",
            f"See {example_url} for more details.",
        ]
        raise ValueError("\n".join(error_lines))


def _validate_fidel_args(continual_eval: bool, fidel_keys: list[str]) -> None:
    # Guarantee the sufficiency: continual_eval ==> len(fidel_keys) == 1
    if continual_eval and len(fidel_keys) != 1:
        raise ValueError(f"continual_max_fidel is valid only if fidel_keys has only one element, but got {fidel_keys=}")


def _validate_output(results: dict[str, float], stored_obj_keys: list[str]) -> None:
    keys_in_output = set(results.keys())
    keys = set(stored_obj_keys)
    if keys_in_output.intersection(keys) != keys:
        raise KeyError(
            f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
            f"but got {results=}"
        )


def _validate_fidels(
    fidels: dict[str, int | float] | None,
    fidel_keys: list[str],
    use_fidel: bool,
    continual_eval: bool,
) -> None:
    if not use_fidel and fidels is not None:
        raise ValueError(
            "Objective function got keyword `fidels`, but fidel_keys was not provided in worker instantiation."
        )
    if use_fidel and fidels is None:
        raise ValueError(
            "Objective function did not get keyword `fidels`, but fidel_keys was provided in worker instantiation."
        )

    if continual_eval:
        return

    fidel_key_set = set(({} if fidels is None else fidels).keys())
    if use_fidel and fidel_key_set != set(fidel_keys):
        raise KeyError(f"The keys in fidels must be identical to {fidel_keys=}, but got {fidels=}")


def _validate_fidels_continual(fidels: dict[str, int | float] | None) -> int:
    if fidels is None or len(fidels.values()) != 1:
        raise ValueError(f"fidels must have only one element when continual_max_fidel is provided, but got {fidels=}")

    fidel = next(iter(fidels.values()))
    if not isinstance(fidel, int):
        raise ValueError(f"Fidelity for continual evaluation must be integer, but got {fidel=}")
    if fidel < 0:
        raise ValueError(f"Fidelity for continual evaluation must be non-negative, but got {fidel=}")

    return fidel
