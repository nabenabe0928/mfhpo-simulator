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
            results.append(pool.apply_async(ObjectiveFuncWorker, kwds=self._wrapper_vars.__dict__))

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
        """The meta-wrapper method of the objective function method in WorkerFunc instances.

        This method recognizes each WorkerFunc by process ID and call the corresponding worker based on the ID.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (dict[str, int | float] | None):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, no-fidelity opt.
            **data_to_scatter (Any):
                Data to scatter across workers.
                For example, when the objective function instance has a large file,
                Dask, which is a typical module for parallel optimization, must serialize/deserialize
                the objective function instances. It causes a significant bottleneck.
                By using dask.scatter, we can avoid this problem and this kwargs serves for this purpose.
                Note that since the handling of parallel workers vary depending on packages,
                users must adapt by themselves.

        Returns:
            results (dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
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
