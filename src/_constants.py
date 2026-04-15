from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Final
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Protocol

    import numpy as np

    class ObjectiveFuncType(Protocol):
        def __call__(
            self,
            eval_config: dict[str, Any],
            *,
            seed: int | None = None,
            **data_to_scatter: Any,
        ) -> dict[str, float]:
            """The prototype of the objective function.

            Args:
                eval_config (dict[str, Any]):
                    The configuration to be used in the objective function.
                seed (int | None):
                    The random seed to be used in the objective function.
                **data_to_scatter (Any):
                    Data to scatter across workers.

            Returns:
                results (dict[str, float]):
                    The results of the objective function given the inputs.
                    It must have `objective metric` and `runtime` at least.
                    Otherwise, any other metrics are optional.
            """
            raise NotImplementedError


NEGLIGIBLE_SEC: Final[float] = 1e-12


@dataclass(frozen=True)
class _ResultData:
    cumtime: float
    eval_config: dict[str, Any]
    results: dict[str, float]
    seed: int | None
    config_id: int | None


@dataclass(frozen=True)
class _WrapperVars:
    n_workers: int
    obj_func: ObjectiveFuncType
    n_actual_evals_in_opt: int
    n_evals: int
    max_total_eval_time: float
    obj_keys: list[str]
    runtime_key: str
    seed: int | None
    store_actual_cumtime: bool
    allow_parallel_sampling: bool
    config_tracking: bool
    expensive_sampler: bool

    def validate(self) -> None:
        if self.n_actual_evals_in_opt < self.n_workers + self.n_evals:
            threshold = self.n_workers + self.n_evals
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
    worker_id: str
    worker_index: int
    rng: np.random.RandomState
    stored_obj_keys: list[str]


class AbstractAskTellOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self) -> tuple[dict[str, Any], int | None]:
        """
        The ask method to sample a configuration using an optimizer.

        Returns:
            (eval_config, config_id):
                * eval_config (dict[str, Any]):
                    The configuration to evaluate.
                * config_id (int | None):
                    The identifier of configuration if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        config_id: int | None = None,
    ) -> None:
        """
        The tell method to register a tuple of configuration and results to an optimizer.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            results (dict[str, float]):
                The dict of the return values from the objective function.
            config_id (int | None):
                The identifier of configuration if needed for continual learning.
        """
        raise NotImplementedError
