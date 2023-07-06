from __future__ import annotations

from benchmark_simulator._constants import _StateType
from benchmark_simulator._secure_proc import (
    _cache_state,
    _delete_state,
    _fetch_cache_states,
)
from benchmark_simulator._utils import _SecureLock

import numpy as np


def _validate_continual_max_fidel(continual_max_fidel: int | None, cls_name: str) -> None:
    if not isinstance(continual_max_fidel, int):  # pragma: no cover
        raise ValueError(f"continual_max_fidel must be int for {cls_name}")


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
