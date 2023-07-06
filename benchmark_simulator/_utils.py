import fcntl
import glob
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterator, TextIO

import numpy as np


def _generate_time_hash() -> str:
    hash = hashlib.sha1()
    # Try to make it as unique as possible! time.time() was not sufficient!
    timehash_pid = str(time.time() + os.getpid() + np.random.random() * 1e9)  # DO NOT MAKE IT SIMPLE
    hash.update(timehash_pid.encode("utf-8"))
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
                except IOError:  # pragma: no cover
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
                except IOError:  # pragma: no cover
                    time.sleep(waiting_time)
                    if time.time() - start >= self.time_limit:
                        raise TimeoutError("Timeout during secure edit. Try again.")
                finally:  # pragma: no cover
                    fcntl.flock(f, fcntl.LOCK_UN)


@dataclass(frozen=True)
class _SecureLockForDistributedSystem:  # pragma: no cover
    """
    If _SecureLock does not work on some environments, we should use it.
    Although this lock still cannot guarantee that it works perfectly, better than nothing.
    Typically, network latency causes some unexpected consequence.

    This class acquires a file lock by publishing a token file.
    If there exists a public token, this class renames it to private token and acquires the access.
    As there is not public token in this situation, other processes cannot access.
    Once the process finishes the access, it renames back to the public token and other processes can access.
    As we acquire the lock directory-wise, it is much slower than with fcntl.
    """

    dir_name: str
    private_token: str
    public_token: str
    token_pattern: str = "access_*.token"
    waiting_time: float = 1e-4
    time_limit: float = 10.0

    def _validate(self) -> None:
        private_token, public_token, token_pattern = self.private_token, self.public_token, self.token_pattern
        if not private_token.startswith(self.dir_name) or not fnmatch(private_token, token_pattern):
            raise ValueError(f"private_token must match the token pattern {token_pattern=}, but got {private_token=}")
        if not public_token.startswith(self.dir_name) or not fnmatch(public_token, token_pattern):
            raise ValueError(f"public_token must match the token pattern {token_pattern=}, but got {public_token=}")

    def init_token(self) -> None:
        self._validate()
        token_pattern = os.path.join(self.dir_name, self.token_pattern)
        n_tokens = len(glob.glob(token_pattern))
        if n_tokens == 0:
            with open(self.public_token, mode="w"):
                pass
        elif n_tokens > 1:  # Token from another process could exist!
            raise FileExistsError

    def _publish_token(self) -> None:
        start = time.time()
        waiting_time = self.waiting_time * (1 + np.random.random())
        while True:
            try:
                os.rename(self.public_token, self.private_token)
                return
            except FileNotFoundError:
                time.sleep(waiting_time)
                if time.time() - start >= self.time_limit:
                    raise TimeoutError("Timeout during token publication. Please remove token files and try again.")

    def _remove_token(self) -> None:
        os.rename(self.private_token, self.public_token)

    @contextmanager
    def read(self, path: str) -> Iterator[TextIO]:
        self._publish_token()
        with open(path, mode="r") as f:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            yield f
        self._remove_token()

    @contextmanager
    def edit(self, path: str) -> Iterator[TextIO]:
        self._publish_token()
        with open(path, mode="r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield f
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
        self._remove_token()
