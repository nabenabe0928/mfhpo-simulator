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
def dummy_edit(f: TextIOWrapper, key: str, num: int) -> None:
    json.load(f)
    f.seek(0)
    json.dump({key: num}, f, indent=4)
    time.sleep(5e-4)


def dummy_reader(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    time.sleep(5e-4)
    return dummy_read(**kwargs)


def dummy_editer(kwargs: Dict[str, Any]) -> None:
    dummy_edit(**kwargs)


def test_secure_read():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    start = time.time()
    pool = multiprocessing.Pool(processes=20)
    for _ in range(10):
        pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0)])
        pool.apply_async(dummy_reader, args=[dict(path=name)])
    pool.close()
    pool.join()
    os.remove(name)
    assert time.time() - start >= 1e-2


def test_secure_read_time_limit():
    name = "test/dummy.json"
    with open(name, mode="w") as f:
        json.dump({}, f)

    n_workers = 2
    pool = multiprocessing.Pool(processes=n_workers)
    pool.apply_async(dummy_editer, args=[dict(path=name, key="a", num=0)])
    r = pool.apply_async(dummy_reader, args=[dict(path=name, time_limit=4e-4)])
    pool.close()
    pool.join()
    with pytest.raises(TimeoutError):
        r.get()

    os.remove(name)


def test_secure_edit():
    pass


if __name__ == "__main__":
    unittest.main()
