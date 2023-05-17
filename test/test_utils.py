import multiprocessing
import os
import pytest
import time
import unittest
from typing import Any, Dict

from _io import TextIOWrapper

import ujson as json

from benchmark_simulator._utils import secure_edit, secure_read


@secure_read
def dummy_read(f: TextIOWrapper) -> Dict[str, Any]:
    return json.load(f)


@secure_edit
def dummy_edit(f: TextIOWrapper, key: str, num: int) -> Dict[str, Any]:
    prev = json.load(f)
    f.seek(0)
    json.dump({key: num}, f, indent=4)
    time.sleep(5e-3)
    return prev


def dummy_reader(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    time.sleep(5e-4)
    return dummy_read(**kwargs)


def dummy_editer(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    dummy_edit(**kwargs)


def test_secure_read():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    start = time.time()
    pool = multiprocessing.Pool(processes=20)
    for _ in range(10):
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0)])
        r.get()  # check error
        r = pool.apply_async(dummy_reader, args=[dict(path=name)])
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
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0)])
        r.get()  # check error
        r = pool.apply_async(dummy_reader, args=[dict(path=name, time_limit=4e-3)])
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
        r = pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0)])
        r.get()  # check error

    pool.close()
    pool.join()
    assert time.time() - start >= 5e-2
    os.remove(name)


if __name__ == "__main__":
    unittest.main()
