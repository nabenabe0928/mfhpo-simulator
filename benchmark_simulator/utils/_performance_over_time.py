from __future__ import annotations

import json
import os

import numpy as np


def _validate_performance(
    cumtimes: np.ndarray | list[np.ndarray] | list[list[float]],
    perf_vals: np.ndarray | list[np.ndarray] | list[list[float]],
    minimize: bool,
) -> np.ndarray:
    n_seeds = len(cumtimes)
    loss_vals = []
    if len(cumtimes) != len(perf_vals):
        raise ValueError(
            "The number of seeds used in cumtimes and perf_vals must be identical, but got "
            f"{len(cumtimes)=} and {len(perf_vals)=}."
        )

    for i in range(n_seeds):
        cumtime, perf_val = cumtimes[i], perf_vals[i]
        if not isinstance(cumtime, (list, np.ndarray)) or not isinstance(perf_val, (list, np.ndarray)):
            raise TypeError(f"cumtimes and perf_vals must be 2D array or list, but got {cumtimes=} and {perf_vals=}.")

        cumtime, perf_val = map(np.asarray, [cumtimes[i], perf_vals[i]])
        if cumtime.shape != perf_val.shape:
            raise ValueError(
                "The shape of each cumtimes and perf_vals for each seed must be identical, but got "
                f"{cumtime.shape=} and {perf_val.shape=}."
            )

        loss_vals.append((2 * minimize - 1) * perf_val.copy())

    return loss_vals


def get_performance_over_time(
    cumtimes: np.ndarray | list[np.ndarray] | list[list[float]],
    perf_vals: np.ndarray | list[np.ndarray] | list[list[float]],
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
    loss_vals = _validate_performance(cumtimes=cumtimes, perf_vals=perf_vals, minimize=minimize)
    n_seeds = len(cumtimes)
    tmin, tmax = np.min([np.min(cumtime) for cumtime in cumtimes]), np.max([np.max(cumtime) for cumtime in cumtimes])
    time_steps = np.exp(np.linspace(np.log(tmin), np.log(tmax), step)) if log else np.linspace(tmin, tmax, step)
    return_perf_vals = []

    for i in range(n_seeds):
        cumtime, loss = cumtimes[i].copy(), np.minimum.accumulate(loss_vals[i])
        cumtime = np.insert(cumtime, 0, 0.0)
        cumtime = np.append(cumtime, np.inf)
        loss = np.insert(loss, 0, np.nan)
        loss = np.append(loss, loss[-1])
        # cumtime[i - 1] < time_step <= cumtime[i]
        indices = np.searchsorted(cumtime, time_steps, side="left")
        return_perf_vals.append((2 * minimize - 1) * loss[indices])

    return time_steps, np.asarray(return_perf_vals)


def get_performance_over_time_from_paths(
    paths: list[str],
    obj_key: str,
    step: int = 100,
    minimize: bool = True,
    log: bool = True,
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
    for path in paths:
        with open(os.path.join(path, "results.json"), mode="r") as f:
            data = json.load(f)
            cumtimes.append(np.asarray(data["cumtime"]))
            perf_vals.append(np.asarray(data[obj_key]))

    time_steps, return_perf_vals = get_performance_over_time(
        cumtimes=cumtimes, perf_vals=perf_vals, minimize=minimize, step=step, log=log
    )
    return time_steps, return_perf_vals
