from __future__ import annotations

import fcntl
import os
import time
import warnings

from _io import TextIOWrapper

from benchmark_simulator._constants import (
    _SharedDataLocations,
    _StateType,
    _TimeStampDictType,
    _TimeValue,
)
from benchmark_simulator._utils import secure_edit, secure_read

import numpy as np

import ujson as json  # type: ignore


def _init_simulator(dir_name: str) -> None:
    for fn in _SharedDataLocations:
        path = os.path.join(dir_name, fn.value)
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
    prev_cumtime = record.get(worker_id, 0.0)
    record[worker_id] = np.clip(cumtime, a_min=prev_cumtime, a_max=_TimeValue.crashed.value)
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
    timestamps = {th: _TimeStampDictType(**ts_dict) for th, ts_dict in json.load(f).items()}
    return timestamps


def _fetch_proc_alloc(path: str) -> dict[int, int]:
    return _complete_proc_allocation(path=path)


@secure_edit
def _record_result(f: TextIOWrapper, results: dict[str, float], fixed: bool = True) -> None:
    record = json.load(f)
    n_observations = len(record.get("cumtime", []))
    keys = list(set(list(record.keys()) + list(results.keys()))) if not fixed else results.keys()
    for key in keys:
        val = results.get(key, None)
        if n_observations == 0:
            record[key] = [val]
        elif key not in record:
            record[key] = [None] * n_observations + [val]
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


def _start_timestamp(path: str, worker_id: str, prev_timestamp: float) -> None:
    _record_timestamp(path=path, worker_id=worker_id, prev_timestamp=time.time(), waited_time=0.0)


def _start_worker_timer(path: str, worker_id: str) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=0.0)


def _finish_worker_timer(path: str, worker_id: str) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=_TimeValue.terminated.value)


def _kill_worker_timer(path: str, worker_id: str) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=_TimeValue.crashed.value)


def _kill_worker_timer_with_min_cumtime(path: str) -> None:
    cumtimes = _fetch_cumtimes(path=path)
    worker_id = min(cumtimes, key=cumtimes.get)
    _kill_worker_timer(path=path, worker_id=worker_id)


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


def _raise_unexpected_timeout_error(max_waiting_time: float) -> None:
    raise TimeoutError(
        f"The simulation was terminated due to too long waiting time (> {max_waiting_time} seconds). \n"
        "You can avoid this error by setting `max_waiting_time=np.inf`,\nbut this is not recommended because "
        "n_workers may not be consistent throughout the simulation without setting max_waiting_time.\n"
        "The possible reasons for the termination are the following:\n"
        "\t1. Your sampler takes too long time to sample for the provided `max_waiting_time`,\n"
        "\t2. n_workers is too large for the provided `max_waiting_time`,\n"
        "\t3. Some workers crashed or new workers were added, and this caused hang, or\n"
        "\t4. The RAM usage is too high due to the benchmark dataset which you use.\n"
        "When you get this error, please report it to our GitHub repository with the following infos:\n"
        "\t1. Your OS and its version,\n"
        "\t2. Python version,\n"
        "\t3. Your optimizer package name and its internal distributed computation module, and\n"
        "\t4. Your benchmark details.\n"
    )


def _terminate_with_unexpected_timeout(path: str, worker_id: str, max_waiting_time: float) -> None:
    _kill_worker_timer(path=path, worker_id=worker_id)
    # The worker with minimum cumlative time may be able to evaluate several HPs during the wait,
    # but it does not matter because the timeout happens due to too long wait.
    time.sleep(1.0)
    _kill_worker_timer_with_min_cumtime(path=path)
    _raise_unexpected_timeout_error(max_waiting_time=max_waiting_time)


def _wait_until_next(
    path: str,
    worker_id: str,
    waiting_time: float = 1e-4,
    warning_interval: int = 10,
    max_waiting_time: float = np.inf,
) -> None:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_min_cumtime(path, worker_id=worker_id):
        time.sleep(waiting_time)
        curtime = time.time()
        if int(curtime - start + 1) % warning_interval == 0:
            warnings.warn(
                "Workers might be hanging. Please consider setting `max_waiting_time` (< np.inf).\n"
                "Note that if samplers or the objective function need long time (> 10 seconds), "
                "please ignore this warning."
            )

        if curtime - start > max_waiting_time:
            _terminate_with_unexpected_timeout(path=path, worker_id=worker_id, max_waiting_time=max_waiting_time)
