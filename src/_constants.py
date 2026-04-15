from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Final
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Protocol

    class ObjectiveFuncType(Protocol):
        def __call__(self, eval_config: dict[str, Any]) -> dict[str, float]:
            """The prototype of the objective function.

            Args:
                eval_config (dict[str, Any]):
                    The configuration to be used in the objective function.

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


@dataclass(frozen=True)
class _WrapperVars:
    n_workers: int
    obj_func: ObjectiveFuncType
    n_actual_evals_in_opt: int
    n_evals: int
    max_total_eval_time: float
    obj_keys: list[str]
    runtime_key: str
    store_actual_cumtime: bool
    allow_parallel_sampling: bool
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


class AbstractAskTellOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self) -> dict[str, Any]:
        """
        The ask method to sample a configuration using an optimizer.

        Returns:
            eval_config (dict[str, Any]):
                The configuration to evaluate.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, eval_config: dict[str, Any], results: dict[str, float]) -> None:
        """
        The tell method to register a tuple of configuration and results to an optimizer.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            results (dict[str, float]):
                The dict of the return values from the objective function.
        """
        raise NotImplementedError
