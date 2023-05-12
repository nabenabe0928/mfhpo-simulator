from typing import Any, Dict, NewType, Optional, Protocol, Tuple


class _ObjectiveFunc(Protocol):
    def __call__(self, config: Dict[str, Any], budget: int, seed: Optional[int] = None) -> Dict[str, float]:
        raise NotImplementedError


DIR_NAME = "mfhpo-simulator-info/"
WORKER_CUMTIME_FILE_NAME = "simulated_cumtime.json"
RESULT_FILE_NAME = "results.json"
PROC_ALLOC_NAME = "proc_alloc.json"
STATE_CACHE_FILE_NAME = "state_cache.json"
INF = 1 << 40
_RuntimeType = NewType("_RuntimeType", float)
_CumtimeType = NewType("_CumtimeType", float)
_BudgetType = NewType("_BudgetType", int)
_SeedType = NewType("_SeedType", Optional[int])
_StateType = Tuple[_RuntimeType, _CumtimeType, _BudgetType, _SeedType]
INIT_STATE: _StateType = [0.0, 0.0, 0, None]
