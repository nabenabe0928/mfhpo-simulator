from __future__ import annotations

import multiprocessing
import os
import pytest
import time
import unittest
from typing import Any

import ujson as json

from benchmark_simulator._utils import _SecureLock


LOCK = _SecureLock()


def dummy_read(path: str, lock: _SecureLock) -> dict[str, Any]:
    with lock.read(path) as f:
        result = json.load(f)

    return result


def dummy_edit(path: str, key: str, num: int, lock: _SecureLock) -> dict[str, Any]:
    with lock.edit(path) as f:
        prev = json.load(f)
        f.seek(0)
        json.dump({key: num}, f, indent=4)
        time.sleep(5e-3)

    return prev


def dummy_reader(kwargs: dict[str, Any]) -> dict[str, Any]:
    time.sleep(5e-4)
    return dummy_read(**kwargs)


def dummy_editer(kwargs: dict[str, Any]) -> dict[str, Any]:
    dummy_edit(**kwargs)


def test_secure_read():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    start = time.time()
    pool = multiprocessing.Pool(processes=20)
    for _ in range(10):
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0, lock=LOCK)])
        r.get()  # check error
        r = pool.apply_async(dummy_reader, args=[dict(path=name, lock=LOCK)])
        r.get()  # check error

    pool.close()
    pool.join()
    os.remove(name)
    assert time.time() - start >= 1e-2


def test_secure_read_time_limit():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    n_workers = 20
    pool = multiprocessing.Pool(processes=n_workers)
    results = []
    for _ in range(n_workers // 2):
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0, lock=LOCK)])
        r.get()  # check error
        r = pool.apply_async(dummy_reader, args=[dict(path=name, lock=_SecureLock(time_limit=4e-3))])
        results.append(r)

    pool.close()
    pool.join()
    with pytest.raises(TimeoutError):
        for r in results:
            r.get()

    os.remove(name)


def test_secure_edit_time_limit():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    n_workers = 10
    pool = multiprocessing.Pool(processes=n_workers)
    start = time.time()
    for _ in range(n_workers):
        # fcntl.flock automatically waits for another worker
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0, lock=LOCK)])
        r.get()  # check error

    pool.close()
    pool.join()
    assert time.time() - start >= 5e-2
    os.remove(name)


if __name__ == "__main__":
    unittest.main()
