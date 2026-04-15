from __future__ import annotations

from src._constants import AbstractAskTellOptimizer


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
