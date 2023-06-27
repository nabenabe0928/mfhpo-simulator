from __future__ import annotations

import os
import threading
from multiprocessing import Pool
from typing import Any

from benchmark_simulator._secure_proc import (
    _allocate_proc_to_worker,
    _fetch_proc_alloc,
    _is_allocation_ready,
    _wait_proc_allocation,
)
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._worker import ObjectiveFuncWorker


class CentralWorkerManager(_BaseWrapperInterface):
    """A central worker manager class.
    This class is supposed to be instantiated if the optimizer module uses multiprocessing.
    For example, Dask, multiprocessing, and joblib would need this class.
    This class recognizes each worker by process ID.
    Therefore, process ID for each worker must be always unique and identical.

    Note:
        See benchmark_simulator/simulator.py to know variables shared across workers.
    """

    def _init_wrapper(self) -> None:
        self._workers: list[ObjectiveFuncWorker]
        self._main_pid = os.getpid()
        self._init_workers()
        self._pid_to_index: dict[int, int] = {}

    def _init_workers(self) -> None:
        if os.path.exists(self.dir_name):
            raise FileExistsError(f"The directory `{self.dir_name}` already exists. Remove it first.")

        pool = Pool()
        results = []
        for _ in range(self._wrapper_vars.n_workers):
            results.append(pool.apply_async(ObjectiveFuncWorker, kwds=dict(wrapper_vars=self._wrapper_vars)))

        pool.close()
        pool.join()
        self._workers = [result.get() for result in results]

    def _init_alloc(self, pid: int) -> None:
        path = self._paths.proc_alloc
        if not _is_allocation_ready(path=path, n_workers=self._wrapper_vars.n_workers, lock=self._lock):
            _allocate_proc_to_worker(path=path, pid=pid, lock=self._lock)
            self._pid_to_index = _wait_proc_allocation(
                path=path, n_workers=self._wrapper_vars.n_workers, lock=self._lock
            )
        else:
            self._pid_to_index = _fetch_proc_alloc(path=path, lock=self._lock)

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        pid = os.getpid()
        pid = threading.get_ident() if pid == self._main_pid else pid
        if len(self._pid_to_index) != self._wrapper_vars.n_workers:
            self._init_alloc(pid)

        if pid not in self._pid_to_index:
            raise ProcessLookupError(
                f"An unknown process/thread with ID {pid} was specified.\n"
                "It is likely that one of the workers crashed and a new worker was added.\n"
                f"However, worker additions are not allowed in {self.__class__.__name__}."
            )

        worker_index = self._pid_to_index[pid]
        results = self._workers[worker_index](eval_config=eval_config, fidels=fidels, **data_to_scatter)
        return results
