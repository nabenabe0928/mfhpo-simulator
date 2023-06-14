import fcntl
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, TextIO

import numpy as np


def _generate_time_hash() -> str:
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    return hash.hexdigest()


@dataclass(frozen=True)
class _SecureLock:
    waiting_time: float = 1e-4
    time_limit: float = 10.0

    @contextmanager
    def read(self, path: str) -> Iterator[TextIO]:
        start = time.time()
        waiting_time = self.waiting_time * (1 + np.random.random())
        fetched = False
        while not fetched:
            with open(path, "r") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    yield f
                    fetched = True
                except IOError:
                    time.sleep(waiting_time)
                    if time.time() - start >= self.time_limit:
                        raise TimeoutError("Timeout during secure read. Try again.")

    @contextmanager
    def edit(self, path: str) -> Iterator[TextIO]:
        start = time.time()
        waiting_time = self.waiting_time * (1 + np.random.random())
        fetched = False
        while not fetched:
            with open(path, "r+") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    yield f
                    f.truncate()
                    fetched = True
                except IOError:
                    time.sleep(waiting_time)
                    if time.time() - start >= self.time_limit:
                        raise TimeoutError("Timeout during secure edit. Try again.")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
