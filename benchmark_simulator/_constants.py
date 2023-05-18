import os
from typing import Any, Dict, NewType, Optional, Protocol, Tuple, TypedDict


class _TimeStampDictType(TypedDict):
    prev_timestamp: float
    waited_time: float


class _ObjectiveFunc(Protocol):
    def __call__(
        self,
        eval_config: Dict[str, Any],
        fidel: Optional[int] = None,
        seed: Optional[int] = None,
        **data_to_scatter: Any,
    ) -> Dict[str, float]:
        """The prototype of the objective function.

        Args:
            eval_config (Dict[str, Any]):
                The configuration to be used in the objective function.
            fidel (Optional[int]):
                The fidelity to be used in the objective function. Typically training epoch in deep learning.
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
            results (Dict[str, float]):
                The results of the objective function given the inputs.
                It must have `objective metric` and `runtime` at least.
                Otherwise, any other metrics are optional.
        """
        raise NotImplementedError


DIR_NAME = "mfhpo-simulator-info/"
WORKER_CUMTIME_FILE_NAME = "simulated_cumtime.json"
RESULT_FILE_NAME = "results.json"
PROC_ALLOC_NAME = "proc_alloc.json"
STATE_CACHE_FILE_NAME = "state_cache.json"
TIMESTAMP_FILE_NAME = "timestamp.json"
INF = 1 << 40
_RuntimeType = NewType("_RuntimeType", float)
_CumtimeType = NewType("_CumtimeType", float)
_FidelityType = NewType("_FidelityType", int)
_SeedType = NewType("_SeedType", Optional[int])  # type: ignore
_StateType = Tuple[_RuntimeType, _CumtimeType, _FidelityType, _SeedType]
INIT_STATE: _StateType = [0.0, 0.0, 0, None]  # type: ignore


def _get_file_paths(dir_name: str) -> Tuple[str, str, str, str, str]:
    return (
        os.path.join(dir_name, PROC_ALLOC_NAME),
        os.path.join(dir_name, RESULT_FILE_NAME),
        os.path.join(dir_name, STATE_CACHE_FILE_NAME),
        os.path.join(dir_name, WORKER_CUMTIME_FILE_NAME),
        os.path.join(dir_name, TIMESTAMP_FILE_NAME),
    )
