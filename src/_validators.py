from __future__ import annotations

from src._constants import AbstractAskTellOptimizer


def _raise_optimizer_init_error() -> None:
    msg = [
        "The initialization of the optimizer must be cheaper than one objective evuation.",
        "In principle, n_workers is too large for the objective to simulate correctly."
        "Please set expensive_sampler=True or a smaller n_workers, or use a cheaper initialization.",
    ]
    raise TimeoutError("\n".join(msg))


def _validate_opt_class(opt: AbstractAskTellOptimizer) -> None:
    if not hasattr(opt, "ask") or not hasattr(opt, "tell"):
        opt_cls = AbstractAskTellOptimizer
        error_lines = [
            "opt must have `ask` and `tell` methods.",
            f"Inherit `{opt_cls.__name__}` and encapsulate your optimizer instance in the child class.",
            "The description of `ask` method is as follows:",
            f"\033[32m{opt_cls.ask.__doc__}\033[0m",
            "The description of `tell` method is as follows:",
            f"\033[32m{opt_cls.tell.__doc__}\033[0m",
        ]
        raise ValueError("\n".join(error_lines))


def _validate_output(results: dict[str, float], stored_obj_keys: list[str]) -> None:
    keys_in_output = set(results.keys())
    keys = set(stored_obj_keys)
    if keys_in_output.intersection(keys) != keys:
        raise KeyError(
            f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
            f"but got {results=}"
        )
