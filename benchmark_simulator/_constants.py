from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol

import numpy as np


DIR_NAME: Final[str] = "mfhpo-simulator-info/"
INF: Final[float] = float(1 << 30)
NEGLIGIBLE_SEC: Final[float] = 1e-12


@dataclass(frozen=True)
class _SampledTimeDictType:
    before_sample: float
    after_sample: float
    worker_index: int


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
    sampled_time: str
    config_tracker: str
    sample_waiting: str

    def __len__(self) -> int:
        return len(self.__dict__)


@dataclass(frozen=True)
class _ResultData:
    cumtime: float
    eval_config: dict[str, Any]
    results: dict[str, float]
    fidels: dict[str, int | float]
    seed: int | None
    prev_fidel: int | None
    config_id: int | None


class _SharedDataFileNames(Enum):
    proc_alloc: str = "proc_alloc.json"
    result: str = "results.json"
    state_cache: str = "state_cache.json"
    worker_cumtime: str = "simulated_cumtime.json"
    timestamp: str = "timestamp.json"
    sampled_time: str = "sampled_time.json"
    config_tracker: str = "config_tracker.json"
    sample_waiting: str = "sample_waiting.json"


@dataclass(frozen=True)
class _TimeValue:
    terminated: float = float(1 << 40)
    crashed: float = float(1 << 41)


@dataclass(frozen=True)
class _WrapperVars:
    save_dir_name: str
    n_workers: int
    obj_func: ObjectiveFuncType
    n_actual_evals_in_opt: int
    n_evals: int
    max_total_eval_time: float
    obj_keys: list[str]
    runtime_key: str
    fidel_keys: list[str] | None
    seed: int | None
    continual_max_fidel: int | None
    max_waiting_time: float
    check_interval_time: float
    store_config: bool
    allow_parallel_sampling: bool
    config_tracking: bool
    expensive_sampler: bool

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
        if self.allow_parallel_sampling and self.expensive_sampler:
            raise ValueError(
                "expensive_sampler and allow_parallel_sampling cannot be True simultaneously.\n"
                "Note that allow_parallel_sampling=True correctly handles expensive samplers"
                " if sampling happens in parallel."
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


class AbstractAskTellOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None, int | None]:
        """
        The ask method to sample a configuration using an optimizer.

        Args:
            None

        Returns:
            (eval_config, fidels) (tuple[dict[str, Any], dict[str, int | float] | None]):
                * eval_config (dict[str, Any]):
                    The configuration to evaluate.
                    The key is the hyperparameter name and its value is the corresponding hyperparameter value.
                    For example, when returning {"alpha": 0.1, "beta": 0.3}, the objective function evaluates
                    the hyperparameter configuration with alpha=0.1 and beta=0.3.
                * fidels (dict[str, int | float] | None):
                    The fidelity parameters to be used for the evaluation of the objective function.
                    If not multi-fidelity optimization, simply return None.
                * config_id (int | None):
                    The identifier of configuration if needed for continual learning.
                    Not used at all when continual_max_fidel=None.
                    As we internally use a hash of eval_config, it may be unstable if eval_config has float.
                    However, even if config_id is not provided, our simulator works without errors
                    although we cannot guarantee that our simulator recognizes the same configs if a users' optimizer
                    slightly changes the content of eval_config.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
    ) -> None:
        """
        The tell method to register for a tuple of configuration, fidelity, and the results to an optimizer.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            results (dict[str, float]):
                The dict of the return values from the objective function.
            fidels (dict[str, Union[float, int] | None):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, we assume that no fidelity is used.
            config_id (int | None):
                The identifier of configuration if needed for continual learning.
                Not used at all when continual_max_fidel=None.
                As we internally use a hash of eval_config, it may be unstable if eval_config has float.
                However, even if config_id is not provided, our simulator works without errors
                although we cannot guarantee that our simulator recognizes the same configs if a users' optimizer
                slightly changes the content of eval_config.

        Returns:
            None
        """
        raise NotImplementedError


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
            fidels (dict[str, Union[float, int] | None):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, we assume that no fidelity is used.
            seed (int | None):
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
