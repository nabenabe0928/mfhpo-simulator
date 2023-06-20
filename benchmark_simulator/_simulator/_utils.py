from __future__ import annotations


def _validate_fidel_args(continual_eval: bool, fidel_keys: list[str]) -> None:
    # Guarantee the sufficiency: continual_eval ==> len(fidel_keys) == 1
    if continual_eval and len(fidel_keys) != 1:
        raise ValueError(f"continual_max_fidel is valid only if fidel_keys has only one element, but got {fidel_keys}")


def _validate_output(results: dict[str, float], stored_obj_keys: list[str]) -> None:
    keys_in_output = set(results.keys())
    keys = set(stored_obj_keys)
    if keys_in_output.intersection(keys) != keys:
        raise KeyError(
            f"The output of objective must be a superset of {list(keys)} specified in obj_keys and runtime_key, "
            f"but got {results}"
        )


def _validate_provided_fidels(fidels: dict[str, int | float] | None) -> int:
    if fidels is None or len(fidels.values()) != 1:
        raise ValueError(f"fidels must have only one element when continual_max_fidel is provided, but got {fidels}")

    fidel = next(iter(fidels.values()))
    if not isinstance(fidel, int):
        raise ValueError(f"Fidelity for continual evaluation must be integer, but got {fidel}")
    if fidel < 0:
        raise ValueError(f"Fidelity for continual evaluation must be non-negative, but got {fidel}")

    return fidel
