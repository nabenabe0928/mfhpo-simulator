import os
import shutil
import unittest

from benchmark_simulator._constants import (
    WORKER_CUMTIME_FILE_NAME,
    RESULT_FILE_NAME,
    STATE_CACHE_FILE_NAME,
    PROC_ALLOC_NAME,
    _get_file_paths,
)
from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _cache_state,
    _complete_proc_allocation,
    _delete_state,
    _fetch_cache_states,
    _fetch_cumtimes,
    _get_worker_id_to_idx,
    _init_simulator,
    _is_allocation_ready,
    _is_min_cumtime,
    _is_simulator_ready,
    _is_simulator_terminated,
    _record_cumtime,
    _record_result,
)

import ujson as json


DIR_NAME = "test/dummy"


def _init_for_tests():
    os.makedirs(DIR_NAME, exist_ok=True)
    _init_simulator(DIR_NAME)


def test_init_simulator():
    _init_for_tests()
    for fn in _get_file_paths(DIR_NAME):
        assert json.load(open(fn)) == {}

    shutil.rmtree(DIR_NAME)


def test_init_simulator_existing():
    os.makedirs(DIR_NAME, exist_ok=True)
    with open(os.path.join(DIR_NAME, RESULT_FILE_NAME), mode="w"):
        pass

    _init_for_tests()
    for fn in [WORKER_CUMTIME_FILE_NAME, RESULT_FILE_NAME, STATE_CACHE_FILE_NAME, PROC_ALLOC_NAME]:
        assert json.load(open(os.path.join(DIR_NAME, fn))) == {}

    shutil.rmtree(DIR_NAME)


def test_allocate_proc_to_worker():
    _init_for_tests()
    path = os.path.join(DIR_NAME, PROC_ALLOC_NAME)
    ans = {}
    for i in range(10):
        _allocate_proc_to_worker(path, pid=i * 100)
        assert _is_allocation_ready(path, n_workers=i + 1)
        ans[i * 100] = i

    assert _complete_proc_allocation(path) == ans
    assert _get_worker_id_to_idx(path) == {str(p): i for i, p in enumerate(ans.keys())}
    shutil.rmtree(DIR_NAME)


def test_record_cumtime():
    _init_for_tests()
    path = os.path.join(DIR_NAME, WORKER_CUMTIME_FILE_NAME)

    ans = {}
    worker_ids = "abcdefghij"
    min_id, min_cumtime = "*", 1000
    n_reg = 0
    for j in range(2):
        for i in range(10):
            worker_id = worker_ids[i]
            cumtime = i + 0.3 * j
            _record_cumtime(path, worker_id=worker_id, cumtime=cumtime)
            ans[worker_id] = cumtime
            if cumtime < min_cumtime:
                min_cumtime = cumtime
                min_id = i

            assert ans == _fetch_cumtimes(path)
            n_reg = max(i + 1, n_reg)
            assert _is_simulator_ready(path, n_workers=n_reg)
            itr = range(10) if j == 1 else range(i + 1)
            for idx in itr:
                worker_id = worker_ids[idx]
                c1 = bool(_is_min_cumtime(path, worker_id=worker_id))
                c2 = bool(min_id == idx)
                assert not (c1 ^ c2)

    shutil.rmtree(DIR_NAME)


def test_cache_state():
    _cache_state


def test_delete_state():
    _delete_state


def test_fetch_cache_states():
    _fetch_cache_states


def test_is_simulator_terminated():
    _is_simulator_terminated


def test_record_result():
    _record_result


if __name__ == "__main__":
    unittest.main()
