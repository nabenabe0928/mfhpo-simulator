from __future__ import annotations

import os
import pytest
import shutil
import unittest

from benchmark_simulator._constants import (
    _SharedDataFileNames,
    _StateType,
    _get_file_paths,
)
from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _cache_state,
    _complete_proc_allocation,
    _delete_state,
    _fetch_cache_states,
    _fetch_cumtimes,
    _get_timeout_message,
    _get_worker_id_to_idx,
    _init_simulator,
    _is_allocation_ready,
    _is_min_cumtime,
    _is_simulator_ready,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
    _wait_all_workers,
    _wait_proc_allocation,
    _wait_until_next,
)
from benchmark_simulator._utils import _SecureLock

import ujson as json


DIR_NAME = "test/dummy"
LOCK = _SecureLock()


def _init_for_tests():
    os.makedirs(DIR_NAME, exist_ok=True)
    _init_simulator(DIR_NAME)


def test_get_timeout_message():
    assert isinstance(_get_timeout_message("dummy1", "dummy2/dummy3.json"), str)


def test_init_simulator():
    _init_for_tests()
    for fn in _get_file_paths(DIR_NAME).__dict__.values():
        assert json.load(open(fn)) == {}

    _init_for_tests()  # check what happens when we already have the files
    shutil.rmtree(DIR_NAME)


def test_init_simulator_existing():
    os.makedirs(DIR_NAME, exist_ok=True)
    with open(os.path.join(DIR_NAME, _SharedDataFileNames.result.value), mode="w"):
        pass

    _init_for_tests()
    for fn in _SharedDataFileNames:
        assert json.load(open(os.path.join(DIR_NAME, fn.value))) == {}

    shutil.rmtree(DIR_NAME)


def test_allocate_proc_to_worker():
    _init_for_tests()
    path = os.path.join(DIR_NAME, _SharedDataFileNames.proc_alloc.value)
    ans = {}
    for i in range(10):
        _allocate_proc_to_worker(path, pid=i * 100, lock=LOCK)
        assert _is_allocation_ready(path, n_workers=i + 1, lock=LOCK)
        ans[i * 100] = i

    assert _complete_proc_allocation(path, lock=LOCK) == ans
    assert _get_worker_id_to_idx(path, lock=LOCK) == {str(p): i for i, p in enumerate(ans.keys())}

    _wait_proc_allocation(path, n_workers=10, lock=LOCK)
    with pytest.raises(TimeoutError):
        _wait_proc_allocation(path, n_workers=11, time_limit=0.1, lock=LOCK)

    shutil.rmtree(DIR_NAME)


def test_record_cumtime():
    _init_for_tests()
    path = os.path.join(DIR_NAME, _SharedDataFileNames.worker_cumtime.value)

    ans = {}
    worker_ids = "abcdefghij"
    min_id, min_cumtime = "*", 1000
    n_reg = 0
    for j in range(2):
        for i in range(10):
            worker_id = worker_ids[i]
            cumtime = i + 0.3 * j
            _record_cumtime(path, worker_id=worker_id, cumtime=cumtime, lock=LOCK)
            ans[worker_id] = cumtime
            if cumtime < min_cumtime:
                min_cumtime = cumtime
                min_id = i

            assert ans == _fetch_cumtimes(path, lock=LOCK)
            n_reg = max(i + 1, n_reg)
            assert _is_simulator_ready(path, n_workers=n_reg, lock=LOCK)
            itr = range(10) if j == 1 else range(i + 1)
            for idx in itr:
                worker_id = worker_ids[idx]
                c1 = bool(_is_min_cumtime(path, worker_id=worker_id, lock=LOCK))
                c2 = bool(min_id == idx)
                assert not (c1 ^ c2)
                if c1:
                    _wait_until_next(path, worker_id=worker_id, lock=LOCK, waiting_time=1e-4)
                else:
                    pass

    _wait_all_workers(path, n_workers=n_reg, lock=LOCK)
    with pytest.raises(TimeoutError):
        _wait_all_workers(path, n_workers=n_reg + 1, time_limit=0.1, lock=LOCK)

    shutil.rmtree(DIR_NAME)


def test_cache_state():
    # _StateType = Tuple[_RuntimeType, _CumtimeType, _FidelityType, _SeedType]
    _init_for_tests()
    path = os.path.join(DIR_NAME, _SharedDataFileNames.state_cache.value)
    cumtime = 0.0
    ans = []
    for update in [False, True]:
        for i in range(10):
            cumtime += i
            state = _StateType(runtime=float(i), cumtime=cumtime, fidel=i, seed=i)
            if update:
                _cache_state(path, config_hash=0, new_state=state, update_index=i, lock=LOCK)
                ans[i] = state
            else:
                _cache_state(path, config_hash=0, new_state=state, lock=LOCK)
                ans.append(state)

            print(ans, _fetch_cache_states(path, config_hash=0, lock=LOCK))
            assert _fetch_cache_states(path, config_hash=0, lock=LOCK) == ans

    for idx in [5, 6, 7, 4, 3, 2, 1, 0, 0]:
        _delete_state(path, config_hash=0, index=idx, lock=LOCK)
        ans.pop(idx)
        assert _fetch_cache_states(path, config_hash=0, lock=LOCK) == ans
    else:
        _delete_state(path, config_hash=0, index=idx, lock=LOCK)
        assert _fetch_cache_states(path, config_hash=0, lock=LOCK) == []

    shutil.rmtree(DIR_NAME)


def test_record_result():
    _init_for_tests()
    path = os.path.join(DIR_NAME, _SharedDataFileNames.result.value)
    ans = {"cumtime": [], "loss": []}
    for i in range(19):
        _record_result(path, results={"loss": i, "cumtime": i}, lock=LOCK)
        ans["loss"].append(i)
        ans["cumtime"].append(i)
        assert not _is_simulator_terminated(path, max_evals=20, lock=LOCK)
        assert json.load(open(path)) == ans
    else:
        _record_result(path, results={"loss": i + 1, "cumtime": i + 1}, lock=LOCK)
        ans["loss"].append(i + 1)
        ans["cumtime"].append(i + 1)
        assert _is_simulator_terminated(path, max_evals=20, lock=LOCK)
        assert json.load(open(path)) == ans

    shutil.rmtree(DIR_NAME)


if __name__ == "__main__":
    unittest.main()