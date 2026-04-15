from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

from src._constants import _WrapperVars
from src._constants import AbstractAskTellOptimizer


class _BaseWrapperInterface(metaclass=ABCMeta):
    """A base wrapper class for the ask-and-tell worker manager.

    Attributes:
        obj_keys (list[str]):
            The objective (or constraint) names that will be stored in the result.
        runtime_key (str):
            The runtime name that will be used for the scheduling.
        fidel_keys (list[str]):
            The fidelity names that will be used in the input `fidels`.
    """

    def __init__(self, wrapper_vars: _WrapperVars):
        self._wrapper_vars = wrapper_vars
        self._obj_keys, self._runtime_key = wrapper_vars.obj_keys, wrapper_vars.runtime_key
        self._fidel_keys = [] if wrapper_vars.fidel_keys is None else wrapper_vars.fidel_keys[:]
        self._init_wrapper()

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

    def simulate(self, opt: AbstractAskTellOptimizer) -> None:
        raise NotImplementedError
