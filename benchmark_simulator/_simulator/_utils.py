from __future__ import annotations

from benchmark_simulator._constants import AbstractAskTellOptimizer


def _raise_optimizer_init_error() -> None:
    msg = [
        "The initialization of the optimizer must be cheaper than one objective evuation.",
        "In principle, n_workers is too large for the objective to simulate correctly."
        "Please set expensive_sampler=True or a smaller n_workers, or use a cheaper initialization.",
    ]
    raise TimeoutError("\n".join(msg))


def _validate_opt_class(opt: AbstractAskTellOptimizer) -> None:
    if not hasattr(opt, "ask") or not hasattr(opt, "tell"):
        example_url = "https://github.com/nabenabe0928/mfhpo-simulator/blob/main/examples/ask_and_tell/"
        opt_cls = AbstractAskTellOptimizer
        error_lines = [
            "opt must have `ask` and `tell` methods.",
            f"Inherit `{opt_cls.__name__}` and encapsulate your optimizer instance in the child class.",
            "The description of `ask` method is as follows:",
            f"\033[32m{opt_cls.ask.__doc__}\033[0m",
            "The description of `tell` method is as follows:",
            f"\033[32m{opt_cls.tell.__doc__}\033[0m",
            f"See {example_url} for more details.",
        ]
        raise ValueError("\n".join(error_lines))


def _validate_fidel_args(continual_eval: bool, fidel_keys: list[str]) -> None:
    # Guarantee the sufficiency: continual_eval ==> len(fidel_keys) == 1
    if continual_eval and len(fidel_keys) != 1:
        raise ValueError(f"continual_max_fidel is valid only if fidel_keys has only one element, but got {fidel_keys=}")


def _validate_output(results: dict[str, float], stored_obj_keys: list[str]) -> None:
    keys_in_output = set(results.keys())
    keys = set(stored_obj_keys)
    if keys_in_output.intersection(keys) != keys:
        raise KeyError(
            f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
            f"but got {results=}"
        )


def _validate_fidels(
    fidels: dict[str, int | float] | None,
    fidel_keys: list[str],
    use_fidel: bool,
    continual_eval: bool,
) -> None:
    if not use_fidel and fidels is not None:
        raise ValueError(
            "Objective function got keyword `fidels`, but fidel_keys was not provided in worker instantiation."
        )
    if use_fidel and fidels is None:
        raise ValueError(
            "Objective function did not get keyword `fidels`, but fidel_keys was provided in worker instantiation."
        )

    if continual_eval:
        return

    fidel_key_set = set(({} if fidels is None else fidels).keys())
    if use_fidel and fidel_key_set != set(fidel_keys):
        raise KeyError(f"The keys in fidels must be identical to {fidel_keys=}, but got {fidels=}")


def _validate_fidels_continual(fidels: dict[str, int | float] | None) -> int:
    if fidels is None or len(fidels.values()) != 1:
        raise ValueError(f"fidels must have only one element when continual_max_fidel is provided, but got {fidels=}")

    fidel = next(iter(fidels.values()))
    if not isinstance(fidel, int):
        raise ValueError(f"Fidelity for continual evaluation must be integer, but got {fidel=}")
    if fidel < 0:
        raise ValueError(f"Fidelity for continual evaluation must be non-negative, but got {fidel=}")

    return fidel
