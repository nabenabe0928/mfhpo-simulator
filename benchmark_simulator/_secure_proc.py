from __future__ import annotations

import fcntl
import os
import time
import warnings
from typing import Any

from benchmark_simulator._constants import (
    _SharedDataFileNames,
    _StateType,
    _TIME_VALUES,
    _TimeNowDictType,
)
from benchmark_simulator._utils import _SecureLock

import numpy as np

import ujson as json  # type: ignore


def _init_simulator(dir_name: str, worker_index: int | None) -> None:
    if worker_index is not None and worker_index != 0:  # only if worker index == 0, we initialize
        return

    # Prevent unnecessary overwrite from any other workers by making workers wait for a random fraction
    time.sleep(np.random.random() * 1e-2)  # DO NOT REMOVE
    for fn in _SharedDataFileNames:
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


def _allocate_proc_to_worker(path: str, pid: int, lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        cur_alloc = json.load(f)
        cur_alloc[pid] = 0
        f.seek(0)
        json.dump(cur_alloc, f, indent=4)


def _complete_proc_allocation(path: str, lock: _SecureLock) -> dict[int, int]:
    with lock.edit(path) as f:
        alloc = json.load(f)
        sorted_pids = np.sort([int(pid) for pid in alloc.keys()])
        alloc = {pid: idx for idx, pid in enumerate(sorted_pids)}
        f.seek(0)
        json.dump(alloc, f, indent=4)

    return alloc


def _record_timenow(path: str, timenow_data: _TimeNowDictType, lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        record = json.load(f)
        for k, v in timenow_data.__dict__.items():
            if k not in record:
                record[k] = [v]
            else:
                record[k].append(v)

        f.seek(0)
        json.dump(record, f, indent=4)


def _record_cumtime(path: str, worker_id: str, cumtime: float, lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        record = json.load(f)
        prev_cumtime = record.get(worker_id, 0.0)
        record[worker_id] = np.clip(cumtime, a_min=prev_cumtime, a_max=_TIME_VALUES.crashed)
        f.seek(0)
        json.dump(record, f, indent=4)


def _record_timestamp(path: str, worker_id: str, prev_timestamp: float, lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        record = json.load(f)
        record[worker_id] = prev_timestamp
        f.seek(0)
        json.dump(record, f, indent=4)


def _record_existing_configs(path: str, config_id_str: str, config: dict[str, Any], lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        existing_configs = json.load(f)
        existing_configs[config_id_str] = config
        f.seek(0)
        json.dump(existing_configs, f, indent=4)


def _cache_state(
    path: str, config_hash: int, new_state: _StateType, lock: _SecureLock, update_index: int | None = None
) -> None:
    with lock.edit(path) as f:
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


def _delete_state(path: str, config_hash: int, index: int, lock: _SecureLock) -> None:
    with lock.edit(path) as f:
        cache = json.load(f)
        config_hash_str = str(config_hash)
        cache[config_hash_str].pop(index)
        if len(cache[config_hash_str]) == 0:
            cache.pop(config_hash_str)

        f.seek(0)
        json.dump(cache, f, indent=4)


def _fetch_cache_states(path: str, config_hash: int, lock: _SecureLock) -> list[_StateType]:
    with lock.read(path) as f:
        states = json.load(f).get(str(config_hash), [])

    return [_StateType(runtime=state[0], cumtime=state[1], fidel=state[2], seed=state[3]) for state in states]


def _fetch_timenow(path: str, lock: _SecureLock) -> dict[str, np.ndarray]:
    with lock.read(path) as f:
        data = {k: np.asarray(v) for k, v in json.load(f).items()}

    return data


def _fetch_cumtimes(path: str, lock: _SecureLock) -> dict[str, float]:
    with lock.read(path) as f:
        cumtimes = json.load(f)

    return cumtimes


def _fetch_timestamps(path: str, lock: _SecureLock) -> dict[str, float]:
    with lock.read(path) as f:
        timestamps = json.load(f)

    return timestamps


def _fetch_existing_configs(path: str, lock: _SecureLock) -> dict[str, dict[str, Any]]:
    with lock.read(path) as f:
        existing_configs = json.load(f)

    return existing_configs


def _fetch_proc_alloc(path: str, lock: _SecureLock) -> dict[int, int]:  # pragma: no cover
    return _complete_proc_allocation(path=path, lock=lock)


def _record_result(path: str, results: dict[str, float], lock: _SecureLock, fixed: bool = True) -> None:
    with lock.edit(path) as f:
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


def _is_simulator_terminated(path: str, max_evals: int, lock: _SecureLock) -> bool:
    with lock.read(path) as f:
        result = len(json.load(f)["cumtime"]) >= max_evals

    return result


def _is_simulator_ready(path: str, n_workers: int, lock: _SecureLock) -> bool:
    with lock.read(path) as f:
        result = len(json.load(f)) == n_workers

    return result


def _is_allocation_ready(path: str, n_workers: int, lock: _SecureLock) -> bool:
    with lock.read(path) as f:
        n_allocs = len(json.load(f))

    if n_allocs > n_workers:  # pragma: no cover
        raise ValueError(_get_timeout_message("the allocation of procs", path))

    return n_allocs == n_workers


def _get_worker_id_to_idx(path: str, lock: _SecureLock) -> dict[str, int]:
    with lock.read(path) as f:
        result = {worker_id: idx for idx, worker_id in enumerate(json.load(f).keys())}

    return result


def _is_min_cumtime(path: str, worker_id: str, lock: _SecureLock) -> bool:
    cumtimes = _fetch_cumtimes(path=path, lock=lock)
    proc_cumtime = cumtimes[worker_id]
    return min(cumtime for cumtime in cumtimes.values()) == proc_cumtime


def _start_timestamp(path: str, worker_id: str, prev_timestamp: float, lock: _SecureLock) -> None:
    _record_timestamp(path=path, worker_id=worker_id, prev_timestamp=time.time(), lock=lock)


def _start_worker_timer(path: str, worker_id: str, lock: _SecureLock) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=0.0, lock=lock)


def _finish_worker_timer(path: str, worker_id: str, lock: _SecureLock) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=_TIME_VALUES.terminated, lock=lock)


def _kill_worker_timer(path: str, worker_id: str, lock: _SecureLock) -> None:
    _record_cumtime(path=path, worker_id=worker_id, cumtime=_TIME_VALUES.crashed, lock=lock)


def _kill_worker_timer_with_min_cumtime(path: str, lock: _SecureLock) -> None:
    cumtimes = _fetch_cumtimes(path=path, lock=lock)
    worker_id = min(cumtimes, key=cumtimes.get)  # type: ignore[arg-type]
    _kill_worker_timer(path=path, worker_id=worker_id, lock=lock)


def _get_timeout_message(cause: str, path: str) -> str:
    dir_name = os.path.join(*path.split("/")[:-1])
    msg = [
        f"Timeout in {cause}. There could be three possible reasons:",
        f"(1) The path {dir_name} already existed before the execution of the program.",
        "(2) n_workers specified in your optimizer and that in the simulator might be different.",
        "(3) launch_multiple_wrappers_from_user_side is incorrectly set.",
    ]
    return "\n".join(msg)


def _wait_proc_allocation(
    path: str, n_workers: int, lock: _SecureLock, waiting_time: float = 1e-2, time_limit: float = 10.0
) -> dict[int, int]:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_allocation_ready(path, n_workers=n_workers, lock=lock):
        time.sleep(waiting_time)
        if time.time() - start >= time_limit:
            raise TimeoutError(_get_timeout_message(cause="the allocation of procs", path=path))

    return _complete_proc_allocation(path, lock=lock)


def _wait_all_workers(
    path: str, n_workers: int, lock: _SecureLock, waiting_time: float = 1e-2, time_limit: float = 10.0
) -> dict[str, int]:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_simulator_ready(path, n_workers=n_workers, lock=lock):
        time.sleep(waiting_time)
        if time.time() - start >= time_limit:
            raise TimeoutError(_get_timeout_message(cause="creating a simulator", path=path))

    return _get_worker_id_to_idx(path, lock=lock)


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


def _terminate_with_unexpected_timeout(path: str, worker_id: str, max_waiting_time: float, lock: _SecureLock) -> None:
    _kill_worker_timer(path=path, worker_id=worker_id, lock=lock)
    # The worker with minimum cumlative time may be able to evaluate several HPs during the wait,
    # but it does not matter because the timeout happens due to too long wait.
    time.sleep(1.0)
    _kill_worker_timer_with_min_cumtime(path=path, lock=lock)
    _raise_unexpected_timeout_error(max_waiting_time=max_waiting_time)


def _wait_until_next(
    path: str,
    worker_id: str,
    lock: _SecureLock,
    waiting_time: float,
    warning_interval: int = 10,
    max_waiting_time: float = np.inf,
) -> None:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while not _is_min_cumtime(path, worker_id=worker_id, lock=lock):
        time.sleep(waiting_time)
        curtime = time.time()
        if int(curtime - start + 1) % warning_interval == 0:
            warnings.warn(
                "Workers might be hanging. Please consider setting `max_waiting_time` (< np.inf).\n"
                "Note that if samplers or the objective function need long time (> 10 seconds), or \n"
                "n_workers is large, please ignore this warning."
            )

        if curtime - start > max_waiting_time:
            _terminate_with_unexpected_timeout(
                path=path, worker_id=worker_id, max_waiting_time=max_waiting_time, lock=lock
            )
