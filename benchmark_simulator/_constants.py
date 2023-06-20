from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol

import numpy as np


DIR_NAME: Final[str] = "mfhpo-simulator-info/"
INF: Final[float] = float(1 << 30)


@dataclass(frozen=True)
class _TimeStampDictType:
    prev_timestamp: float
    waited_time: float


@dataclass(frozen=True)
class _TimeNowDictType:
    before_sample: float
    after_sample: float


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
    timenow: str


class _SharedDataFileNames(Enum):
    proc_alloc: str = "proc_alloc.json"
    result: str = "results.json"
    state_cache: str = "state_cache.json"
    worker_cumtime: str = "simulated_cumtime.json"
    timestamp: str = "timestamp.json"
    timenow: str = "timenow.json"


@dataclass(frozen=True)
class _TimeValue:
    terminated: float = float(1 << 40)
    crashed: float = float(1 << 41)


@dataclass(frozen=True)
class _WrapperVars:
    subdir_name: str
    n_workers: int
    obj_func: ObjectiveFuncType
    n_actual_evals_in_opt: int
    n_evals: int
    obj_keys: list[str]
    runtime_key: str
    fidel_keys: list[str] | None
    seed: int | None
    continual_max_fidel: int | None
    max_waiting_time: float
    check_interval_time: float
    store_config: bool

    def validate(self) -> None:
        if self.n_actual_evals_in_opt < self.n_workers + self.n_evals:
            threshold = self.n_workers + self.n_evals
            # In fact, n_workers + n_evals - 1 is the real minimum threshold.
            raise ValueError(
                "Cannot guarantee that optimziers will not hang. "
                f"Use n_actual_evals_in_opt >= {threshold} (= n_evals + n_workers) at least. "
                "Note that our package cannot change your optimizer setting, so "
                "make sure that you changed your optimizer setting, but not only `n_actual_evals_in_opt`."
            )


@dataclass(frozen=True)
class _WorkerVars:
    continual_eval: bool
    worker_id: str
    worker_index: int
    rng: np.random.RandomState
    use_fidel: bool
    stored_obj_keys: list[str]


_TIME_VALUES = _TimeValue()


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
    return _InfoPaths(**{fn.name: os.path.join(dir_name, fn.value) for fn in _SharedDataFileNames})
