from __future__ import annotations

import fcntl
import os
import time
import warnings

from _io import TextIOWrapper

from benchmark_simulator._constants import (
    PROC_ALLOC_NAME,
    RESULT_FILE_NAME,
    STATE_CACHE_FILE_NAME,
    TIMESTAMP_FILE_NAME,
    WORKER_CUMTIME_FILE_NAME,
    _StateType,
    _TimeStampDictType,
)
from benchmark_simulator._utils import secure_edit, secure_read

import numpy as np

import ujson as json  # type: ignore


def _init_simulator(dir_name: str) -> None:
    for fn in [WORKER_CUMTIME_FILE_NAME, RESULT_FILE_NAME, STATE_CACHE_FILE_NAME, PROC_ALLOC_NAME, TIMESTAMP_FILE_NAME]:
        path = os.path.join(dir_name, fn)
        with open(path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            content = f.read()
            if len(content) < 2:
                f.seek(0)
                f.truncate()
                f.write("{}")

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@secure_edit
def _allocate_proc_to_worker(f: TextIOWrapper, pid: int) -> None:
    cur_alloc = json.load(f)
    cur_alloc[pid] = 0
    f.seek(0)
    json.dump(cur_alloc, f, indent=4)


@secure_edit
def _complete_proc_allocation(f: TextIOWrapper) -> dict[int, int]:
    alloc = json.load(f)
    sorted_pids = np.sort([int(pid) for pid in alloc.keys()])
    alloc = {pid: idx for idx, pid in enumerate(sorted_pids)}
    f.seek(0)
    json.dump(alloc, f, indent=4)
    return alloc


@secure_edit
def _record_cumtime(f: TextIOWrapper, worker_id: str, cumtime: float) -> None:
    record = json.load(f)
    record[worker_id] = cumtime
    f.seek(0)
    json.dump(record, f, indent=4)


@secure_edit
def _record_timestamp(f: TextIOWrapper, worker_id: str, prev_timestamp: float, waited_time: float) -> None:
    record = json.load(f)
    record[worker_id] = dict(prev_timestamp=prev_timestamp, waited_time=waited_time)
    f.seek(0)
    json.dump(record, f, indent=4)


@secure_edit
def _cache_state(f: TextIOWrapper, config_hash: int, new_state: _StateType, update_index: int | None = None) -> None:
    config_hash_str = str(config_hash)
    cache = json.load(f)
    _new_state = [new_state.runtime, new_state.cumtime, new_state.fidel, new_state.seed]
    if config_hash_str not in cache:
        cache[config_hash_str] = [_new_state]
    elif update_index is not None:
        cache[config_hash_str][update_index] = _new_state
    else:
        cache[config_hash_str].append(_new_state)

    f.seek(0)
    json.dump(cache, f, indent=4)


@secure_edit
def _delete_state(f: TextIOWrapper, config_hash: int, index: int) -> None:
    cache = json.load(f)
    config_hash_str = str(config_hash)
    cache[config_hash_str].pop(index)
    if len(cache[config_hash_str]) == 0:
        cache.pop(config_hash_str)

    f.seek(0)
    json.dump(cache, f, indent=4)


@secure_read
def _fetch_cache_states(f: TextIOWrapper, config_hash: int) -> list[_StateType]:
    states = json.load(f).get(str(config_hash), [])
    return [_StateType(runtime=state[0], cumtime=state[1], fidel=state[2], seed=state[3]) for state in states]


@secure_read
def _fetch_cumtimes(f: TextIOWrapper) -> dict[str, float]:
    cumtimes = json.load(f)
    return cumtimes


@secure_read
def _fetch_timestamps(f: TextIOWrapper) -> dict[str, _TimeStampDictType]:
    timestamps = json.load(f)
    return timestamps


@secure_edit
def _record_result(f: TextIOWrapper, results: dict[str, float]) -> None:
    record = json.load(f)
    for key, val in results.items():
        if key not in record:
            record[key] = [val]
        else:
            record[key].append(val)

    f.seek(0)
    json.dump(record, f, indent=4)


@secure_read
def _is_simulator_terminated(f: TextIOWrapper, max_evals: int) -> bool:
    return len(json.load(f)["cumtime"]) >= max_evals


@secure_read
def _is_simulator_ready(f: TextIOWrapper, n_workers: int) -> bool:
    return len(json.load(f)) == n_workers


@secure_read
def _is_allocation_ready(f: TextIOWrapper, n_workers: int) -> bool:
    return len(json.load(f)) == n_workers


@secure_read
def _get_worker_id_to_idx(f: TextIOWrapper) -> dict[str, int]:
    return {worker_id: idx for idx, worker_id in enumerate(json.load(f).keys())}


def _is_min_cumtime(path: str, worker_id: str) -> bool:
    cumtimes = _fetch_cumtimes(path=path)
    proc_cumtime = cumtimes[worker_id]
    return min(cumtime for cumtime in cumtimes.values()) == proc_cumtime


def _get_timeout_message(cause: str, path: str) -> str:
    dir_name = os.path.join(*path.split("/")[:-1])
    msg = f"Timeout in {cause}. There could be two possible reasons:\n"
    msg += f"(1) The path {dir_name} already existed before the execution of the program.\n"
    msg += "(2) n_workers specified in your optimizer and that in the simulator might be different."
    return msg


def _wait_proc_allocation(
    path: str, n_workers: int, waiting_time: float = 1e-2, time_limit: float = 10.0
) -> dict[int, int]:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_allocation_ready(path, n_workers=n_workers):
        time.sleep(waiting_time)
        if time.time() - start >= time_limit:
            raise TimeoutError(_get_timeout_message(cause="the allocation of procs", path=path))

    return _complete_proc_allocation(path)


def _wait_all_workers(
    path: str, n_workers: int, waiting_time: float = 1e-2, time_limit: float = 10.0
) -> dict[str, int]:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_simulator_ready(path, n_workers=n_workers):
        time.sleep(waiting_time)
        if time.time() - start >= time_limit:
            raise TimeoutError(_get_timeout_message(cause="creating a simulator", path=path))

    return _get_worker_id_to_idx(path)


def _wait_until_next(path: str, worker_id: str, waiting_time: float = 1e-4, warning_interval: int = 10) -> None:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_min_cumtime(path, worker_id=worker_id):
        time.sleep(waiting_time)
        if int(time.time() - start + 1) % warning_interval == 0:
            warnings.warn(
                "Workers might be hanging. Please make sure `n_evals` is smaller than the actual n_evals.\n"
                "Note that if samplers or the objective function need long time (> 10 seconds), "
                "please ignore this warning."
            )
