import fcntl
import hashlib
import time
from typing import Any, Callable

import numpy as np


def _generate_time_hash() -> str:
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    return hash.hexdigest()


def secure_read(func: Callable) -> Callable:
    def _inner(path: str, waiting_time: float = 1e-4, time_limit: float = 10.0, **kwargs: Any) -> Any:
        start = time.time()
        waiting_time *= 1 + np.random.random()
        fetched, output = False, None
        while not fetched:
            with open(path, "r") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    output = func(f, **kwargs)
                    fetched = True
                except IOError:
                    time.sleep(waiting_time)
                    if time.time() - start >= time_limit:
                        raise TimeoutError("Timeout during secure read. Try again.")

        return output

    return _inner


def secure_edit(func: Callable) -> Callable:
    def _inner(path: str, waiting_time: float = 1e-4, time_limit: float = 10.0, **kwargs: Any) -> Any:
        start = time.time()
        waiting_time *= 1 + np.random.random()
        fetched, output = False, None
        while not fetched:
            with open(path, "r+") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    output = func(f, **kwargs)
                    f.truncate()
                    fetched = True
                except IOError:
                    time.sleep(waiting_time)
                    if time.time() - start >= time_limit:
                        raise TimeoutError("Timeout during secure edit. Try again.")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        return output

    return _inner
