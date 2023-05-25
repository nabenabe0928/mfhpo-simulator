from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypedDict


class _TimeStampDictType(TypedDict):
    prev_timestamp: float
    waited_time: float


@dataclass(frozen=True)
class _StateType:
    runtime: float = 0.0
    cumtime: float = 0.0
    fidel: int = 0
    seed: int | None = None


class ObjectiveFuncType(Protocol):
    def __call__(
        self,
        eval_config: dict[str, Any],
        fidels: dict[str, int | float] | None = None,
        seed: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        """The prototype of the objective function.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            fidels (Optional[dict[str, Union[float, int]]):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, we assume that no fidelity is used.
            seed (Optional[int]):
                The random seed to be used in the objective function.
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
        raise NotImplementedError


DIR_NAME: Final[str] = "mfhpo-simulator-info/"
WORKER_CUMTIME_FILE_NAME: Final[str] = "simulated_cumtime.json"
RESULT_FILE_NAME: Final[str] = "results.json"
PROC_ALLOC_NAME: Final[str] = "proc_alloc.json"
STATE_CACHE_FILE_NAME: Final[str] = "state_cache.json"
TIMESTAMP_FILE_NAME: Final[str] = "timestamp.json"
INF: Final[int] = 1 << 40


def _get_file_paths(dir_name: str) -> tuple[str, str, str, str, str]:
    return (
        os.path.join(dir_name, PROC_ALLOC_NAME),
        os.path.join(dir_name, RESULT_FILE_NAME),
        os.path.join(dir_name, STATE_CACHE_FILE_NAME),
        os.path.join(dir_name, WORKER_CUMTIME_FILE_NAME),
        os.path.join(dir_name, TIMESTAMP_FILE_NAME),
    )
