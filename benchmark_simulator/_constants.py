from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol


DIR_NAME: Final[str] = "mfhpo-simulator-info/"
INF: Final[float] = float(1 << 30)


@dataclass(frozen=True)
class _TimeStampDictType:
    prev_timestamp: float
    waited_time: float


@dataclass(frozen=True)
class _StateType:
    runtime: float = 0.0
    cumtime: float = 0.0
    fidel: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class _InfoPaths:
    proc_alloc: str
    result: str
    state_cache: str
    worker_cumtime: str
    timestamp: str


class _SharedDataLocations(Enum):
    proc_alloc: str = "proc_alloc.json"
    result: str = "results.json"
    state_cache: str = "state_cache.json"
    worker_cumtime: str = "simulated_cumtime.json"
    timestamp: str = "timestamp.json"


class _TimeValue(Enum):
    terminated: float = float(1 << 40)
    crashed: float = float(1 << 41)


class ObjectiveFuncType(Protocol):
    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
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


def _get_file_paths(dir_name: str) -> _InfoPaths:
    return _InfoPaths(**{fn.name: os.path.join(dir_name, fn.value) for fn in _SharedDataLocations})
