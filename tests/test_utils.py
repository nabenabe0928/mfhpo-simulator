from __future__ import annotations

import os
import pytest
import time
import unittest
from contextlib import contextmanager
from typing import Any

import ujson as json

from benchmark_simulator._utils import _SecureLock

from tests.utils import get_pool


LOCK = _SecureLock()
PATH = "tests/dummy.json"


def dummy_read(lock: _SecureLock) -> dict[str, Any]:
    time.sleep(5e-4)
    with lock.read(PATH) as f:
        result = json.load(f)

    return result


def dummy_edit(lock: _SecureLock) -> dict[str, Any]:
    with lock.edit(PATH) as f:
        prev = json.load(f)
        f.seek(0)
        json.dump({"a": 0}, f, indent=4)
        time.sleep(5e-3)

    return prev


@contextmanager
def common_proc(n_workers: int):
    with open(PATH, mode="w") as f:
        json.dump({}, f)

    with get_pool(n_workers=n_workers) as pool:
        yield pool

    os.remove(PATH)


def test_secure_read():
    n_workers = 20
    with common_proc(n_workers=n_workers) as pool:
        start = time.time()
        for _ in range(n_workers // 2):
            r = pool.apply_async(dummy_edit, args=[LOCK])
            r.get()  # check error
            r = pool.apply_async(dummy_read, args=[LOCK])
            r.get()  # check error

    assert time.time() - start >= 1e-2


def test_secure_read_time_limit():
    n_workers = 20
    results = []
    with common_proc(n_workers=n_workers) as pool:
        for _ in range(n_workers // 2):
            r = pool.apply_async(dummy_edit, args=[LOCK])
            r.get()  # check error
            r = pool.apply_async(dummy_read, args=[_SecureLock(time_limit=4e-3)])
            results.append(r)

    with pytest.raises(TimeoutError):
        for r in results:
            r.get()


def test_secure_edit_time_limit():
    n_workers = 10
    with common_proc(n_workers=n_workers) as pool:
        start = time.time()
        for _ in range(n_workers):
            # fcntl.flock automatically waits for another worker
            r = pool.apply_async(dummy_edit, args=[LOCK])
            r.get()  # check error

    assert time.time() - start >= 5e-2


if __name__ == "__main__":
    unittest.main()
