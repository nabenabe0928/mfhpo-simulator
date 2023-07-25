from __future__ import annotations

import json
import os

import numpy as np


def _validate_performance(
    cumtimes: np.ndarray | list[np.ndarray] | list[list[float]],
    perf_vals: np.ndarray | list[np.ndarray] | list[list[float]],
    optimizer_overheads: np.ndarray | list[np.ndarray] | list[list[float]],
    minimize: bool,
) -> np.ndarray:
    n_seeds = len(cumtimes)
    loss_vals = []
    if len(cumtimes) != len(perf_vals) or len(cumtimes) != len(optimizer_overheads):
        raise ValueError(
            "The number of seeds used in cumtimes, perf_vals, and optimizer_overheads must be identical, but got "
            f"{len(cumtimes)=}, {len(perf_vals)=}, and {len(optimizer_overheads)=}."
        )

    for i in range(n_seeds):
        cumtime, perf_val, opt_overhead = cumtimes[i], perf_vals[i], optimizer_overheads[i]
        if any(not isinstance(obj, (list, np.ndarray)) for obj in [cumtime, perf_val, opt_overhead]):
            raise TypeError(
                "cumtimes, perf_vals, and optimizer_overheads must be 2D array or list, but got "
                f"{cumtimes=}, {perf_vals=}, and {optimizer_overheads=}."
            )

        cumtime, perf_val, opt_overhead = map(np.asarray, [cumtimes[i], perf_vals[i], optimizer_overheads[i]])
        if cumtime.shape != perf_val.shape or cumtime.shape != opt_overhead.shape:
            raise ValueError(
                "The shapes of cumtimes, perf_vals, and optimizer_overheads for each seed must be identical, but got "
                f"{cumtime.shape=}, {perf_val.shape=}, {opt_overhead.shape=}."
            )
        if np.any(cumtime <= opt_overhead):
            raise ValueError("Each element of optimizer_overheads must be smaller than that of cumtimes.")

        loss_vals.append((2 * minimize - 1) * perf_val.copy())

    return loss_vals


def get_performance_over_time(
    cumtimes: np.ndarray | list[np.ndarray] | list[list[float]],
    perf_vals: np.ndarray | list[np.ndarray] | list[list[float]],
    optimizer_overheads: np.ndarray | list[np.ndarray] | list[list[float]] | None = None,
    step: int = 100,
    minimize: bool = True,
    log: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get performance curve over time across multiple random seeds.
    Since each result has different timepoints for each evaluation, it is cumbersome to compute mean and standard error.
    This function provides a handy way to extract the trajectory at a fixed set of time points.

    Args:
        cumtimes (np.ndarray | list[np.ndarray] | list[list[float]]):
            The cumulative times of each evaluation finished.
            The shape should be (n_seeds, n_evals).
            However, if each seed has different n_evals, users can simply provide a list of arrays with different size.
        perf_vals (np.ndarray | list[np.ndarray] | list[list[float]]):
            The performance metric values of each evaluation.
            The shape should be (n_seeds, n_evals).
            However, if each seed has different n_evals, users can simply provide a list of arrays with different size.
        optimizer_overheads (np.ndarray | list[np.ndarray] | list[list[float]] | None):
            The overheads of optimizer during each optimization.
            The shape should be (n_seeds, n_evals).
            If None is provided, we subduct the overheads from the cumulative time.
        step (int):
            The number of time points to take.
            The minimum/maximum time points are determined based on the provided cumtimes.
                * minimum time points := np.min(cumtimes)
                * maximum time points := np.max(cumtimes)
        minimize (bool):
            Whether the performance metric is better when it is smaller.
            The returned perf_vals will be an increasing sequence if minimize=False.
        log (bool):
            Whether the time points should be taken on log-scale.

    Returns:
        time_steps, perf_vals (tuple[np.ndarray, np.ndarray]):
            time_steps (np.ndarray):
                The time points that were used to extract the perf_vals.
                The shape is (step, ).
            perf_vals (np.ndarray):
                The cumulative best performance metric value up to the corresponding time point.
                The shape is (step, ).
    """
    optimizer_overheads = [np.zeros_like(t) for t in cumtimes] if optimizer_overheads is None else optimizer_overheads
    loss_vals = _validate_performance(
        cumtimes=cumtimes, perf_vals=perf_vals, optimizer_overheads=optimizer_overheads, minimize=minimize
    )
    _cumtimes = [c - o for c, o in zip(cumtimes, optimizer_overheads)]
    n_seeds = len(_cumtimes)
    tmin, tmax = np.min([np.min(cumtime) for cumtime in _cumtimes]), np.max([np.max(cumtime) for cumtime in _cumtimes])
    time_steps = np.exp(np.linspace(np.log(tmin), np.log(tmax), step)) if log else np.linspace(tmin, tmax, step)
    return_perf_vals = []

    for i in range(n_seeds):
        cumtime, loss = _cumtimes[i], np.minimum.accumulate(loss_vals[i])
        cumtime = np.insert(cumtime, 0, 0.0)
        cumtime = np.append(cumtime, np.inf)
        loss = np.insert(loss, 0, np.nan)
        loss = np.append(loss, loss[-1])
        # cumtime[i - 1] < time_step <= cumtime[i]
        indices = np.searchsorted(cumtime, time_steps, side="left")
        return_perf_vals.append((2 * minimize - 1) * loss[indices])

    return time_steps, np.asarray(return_perf_vals)


def _sort_optimizer_overhead(
    optimizer_overhead: np.ndarray,
    saved_worker_indices: np.ndarray,
    correct_worker_indices: np.ndarray,
) -> np.ndarray:
    max_worker_index = np.max(correct_worker_indices)
    sorted_optimizer_overhead = np.zeros_like(correct_worker_indices, dtype=np.float64)
    indices = np.arange(optimizer_overhead.size)
    for idx in range(max_worker_index + 1):
        flag = correct_worker_indices == idx
        corresponding_indices = indices[saved_worker_indices == idx]
        sorted_optimizer_overhead[flag] = optimizer_overhead[corresponding_indices][: np.sum(flag)]

    return sorted_optimizer_overhead


def get_performance_over_time_from_paths(
    paths: list[str],
    obj_key: str,
    step: int = 100,
    minimize: bool = True,
    log: bool = True,
    consider_optimizer_overhead: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get performance curve over time across multiple random seeds.
    Since each result has different timepoints for each evaluation, it is cumbersome to compute mean and standard error.
    This function provides a handy way to extract the trajectory at a fixed set of time points.

    Args:
        paths (list[str]):
            The paths of the data to be accounted.
            Since we bundle the information, these paths should contain the results on the same setup
            with different random seeds.
        obj_key (str):
            The key of the performance metric in results.json.
        step (int):
            The number of time points to take.
            The minimum/maximum time points are determined based on the provided cumtimes.
                * minimum time points := np.min(cumtimes)
                * maximum time points := np.max(cumtimes)
        minimize (bool):
            Whether the performance metric is better when it is smaller.
            The returned perf_vals will be an increasing sequence if minimize=False.
        log (bool):
            Whether the time points should be taken on log-scale.
        consider_optimizer_overhead (bool):
            Whetehr to consider the optimizer overhead into the simulated time.
            If False, we will remove the optimizer overhead from the runtime.

    Returns:
        time_steps, perf_vals (tuple[np.ndarray, np.ndarray]):
            time_steps (np.ndarray):
                The time points that were used to extract the perf_vals.
                The shape is (step, ).
            perf_vals (np.ndarray):
                The cumulative best performance metric value up to the corresponding time point.
                The shape is (step, ).
    """
    cumtimes, perf_vals = [], []
    optimizer_overheads: list[np.ndarray] | None = None if consider_optimizer_overhead else []
    for path in paths:
        n_evals = 0
        correct_worker_indices: np.ndarray
        with open(os.path.join(path, "results.json"), mode="r") as f:
            data = json.load(f)
            cumtimes.append(np.asarray(data["cumtime"]))
            n_evals = cumtimes[-1].size
            perf_vals.append(np.asarray(data[obj_key]))
            correct_worker_indices = np.asarray(data["worker_index"])

        if optimizer_overheads is None:
            continue

        with open(os.path.join(path, "sampled_time.json"), mode="r") as f:
            data = json.load(f)
            optimizer_overhead = _sort_optimizer_overhead(
                optimizer_overhead=np.array(data["after_sample"]) - np.array(data["before_sample"]),
                correct_worker_indices=correct_worker_indices,
                saved_worker_indices=np.asarray(data["worker_index"]),
            )
            optimizer_overheads.append(optimizer_overhead[:n_evals])

    time_steps, return_perf_vals = get_performance_over_time(
        cumtimes=cumtimes,
        perf_vals=perf_vals,
        optimizer_overheads=optimizer_overheads,
        minimize=minimize,
        step=step,
        log=log,
    )
    return time_steps, return_perf_vals


def get_mean_and_standard_error(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"The type of the input must be np.ndarray, but got {type(x)}")
    if len(x.shape) != 2:
        raise ValueError(f"The shape of the input array must be 2D, but got {len(x.shape)}D")

    mean = np.nanmean(x, axis=0)
    ste = np.nanstd(x, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(x), axis=0))
    return mean, ste
