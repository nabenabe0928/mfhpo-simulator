from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from typing import Any

from benchmark_simulator._constants import AbstractAskTellOptimizer, DIR_NAME, _WrapperVars, _get_file_paths
from benchmark_simulator._utils import _SecureLock


class _BaseWrapperInterface(metaclass=ABCMeta):
    """A base wrapper class for each worker or manager.
    This wrapper class serves the shared interface of worker and manager class.

    Attributes:
        dir_name (str):
            The directory name where all the information will be stored.
        obj_keys (list[str]):
            The objective (or constraint) names that will be stored in the result file.
        runtime_key (str):
            The runtime name that will be used for the scheduling.
        fidel_keys (list[str]):
            The fidelity names that will be used in the input `fidels`.
    """

    def __init__(self, wrapper_vars: _WrapperVars):
        self._wrapper_vars = wrapper_vars

        self._lock = _SecureLock()
        self._dir_name = os.path.join(DIR_NAME, wrapper_vars.save_dir_name)
        self._paths = _get_file_paths(self.dir_name)
        self._obj_keys, self._runtime_key = wrapper_vars.obj_keys, wrapper_vars.runtime_key
        self._fidel_keys = [] if wrapper_vars.fidel_keys is None else wrapper_vars.fidel_keys[:]
        self._init_wrapper()

    @property
    def dir_name(self) -> str:
        return self._dir_name

    @property
    def obj_keys(self) -> list[str]:
        return self._obj_keys[:]

    @property
    def runtime_key(self) -> str:
        return self._runtime_key

    @property
    def fidel_keys(self) -> list[str]:
        return self._fidel_keys[:]

    @abstractmethod
    def _init_wrapper(self) -> None:
        raise NotImplementedError

    def __call__(
        self,
        eval_config: dict[str, Any],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
        **data_to_scatter: Any,
    ) -> dict[str, float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} is an optimizer wrapper, but not a function wrapper.\n"
            "It is not supposed to be called. If you want to use a function wrapper set `ask_and_tell = False`."
        )

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a function wrapper, but not an optimizer wrapper.\n"
            "If you want to use the ask-and-tell optimizer wrapper set `ask_and_tell = True`."
        )
