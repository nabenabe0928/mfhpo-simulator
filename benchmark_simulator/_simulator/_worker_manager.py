from __future__ import annotations

import os
import threading
import time
from typing import Any

from benchmark_simulator._constants import _WrapperVars
from benchmark_simulator._secure_proc import _allocate_proc_to_worker, _fetch_proc_alloc, _is_allocation_ready
from benchmark_simulator._simulator._base_wrapper import _BaseWrapperInterface
from benchmark_simulator._simulator._worker import _ObjectiveFuncWorker


class _CentralWorkerManager(_BaseWrapperInterface):
    """A central worker manager class.
    This class is supposed to be instantiated if the optimizer module uses multiprocessing.
    For example, Dask, multiprocessing, and joblib would need this class.
    This class recognizes each worker by process ID.
    Therefore, process ID for each worker must be always unique and identical.

    Note:
        See benchmark_simulator/simulator.py to know variables shared across workers.
    """

    def __init__(self, wrapper_vars: _WrapperVars, careful_init: bool):
        self._careful_init = careful_init
        super().__init__(wrapper_vars=wrapper_vars)

    def _init_wrapper(self) -> None:
        self._workers: list[_ObjectiveFuncWorker]
        self._main_pid = os.getpid()
        self._n_workers = self._wrapper_vars.n_workers
        self._init_workers()
        self._pid_to_index: dict[int, int] = {}

    def _init_workers(self) -> None:
        if os.path.exists(self.dir_name):
            raise FileExistsError(f"The directory `{self.dir_name}` already exists. Remove it first.")

        n_workers = self._wrapper_vars.n_workers
        self._workers = [
            _ObjectiveFuncWorker(wrapper_vars=self._wrapper_vars, worker_index=i, async_instantiations=False)
            for i in range(n_workers)
        ]

    def _init_alloc(self, pid: int) -> None:
        path = self._paths.proc_alloc
        if not _is_allocation_ready(path=path, n_workers=self._wrapper_vars.n_workers, lock=self._lock):
            # _allocate_proc_to_worker will not be called twice by each process due to wait_other_workers
            worker_index = _allocate_proc_to_worker(path=path, pid=pid, time_ns=time.time_ns(), lock=self._lock)
            # Important to match the init eval order. The longest latency has 2 * self._n_workers * waiting_time
            waiting_time = 1e-4
            time.sleep(self._careful_init * 2 * self._n_workers * waiting_time * worker_index)
            self._pid_to_index = {pid: worker_index}
        else:
            # This line is actually covered, but it is not visible due to multiprocessing nature
            self._pid_to_index = _fetch_proc_alloc(path=path, lock=self._lock)  # pragma: no cover

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        pid = os.getpid()
        pid = threading.get_ident() if pid == self._main_pid else pid
        if len(self._pid_to_index) != self._n_workers:
            self._init_alloc(pid)

        if pid not in self._pid_to_index:  # pragma: no cover
            raise ProcessLookupError(
                f"An unknown process/thread with ID {pid} was specified.\n"
                "It is likely that one of the workers crashed and a new worker was added or \n"
                f"n_workers in ObjectiveFuncWrapper and your optimizer were incompatible.\n"
                f"However, worker additions are not allowed in ObjectiveFuncWrapper."
            )

        worker_index = self._pid_to_index[pid]
        results = self._workers[worker_index](
            eval_config=eval_config, fidels=fidels, config_id=config_id, **data_to_scatter
        )
        return results
