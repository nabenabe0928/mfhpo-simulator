from __future__ import annotations

import unittest

import numpy as np
import optuna

from src import AsyncOptBenchmarkSimulator
from src.tests.utils import CounterSampler
from src.tests.utils import dummy_no_fidel_func
from src.tests.utils import get_overhead_from_study
from src.tests.utils import get_results_from_study
from src.tests.utils import simplest_dummy_func
from src.tests.utils import TestProblem


DEFAULT_KWARGS = dict(
    n_workers=1,
    n_trials=10,
)

DUMMY_SEARCH_SPACE = {"x": optuna.distributions.IntDistribution(0, 99)}


def _create_study() -> optuna.Study:
    return optuna.create_study(sampler=CounterSampler())


def _create_problem(obj_func=dummy_no_fidel_func) -> TestProblem:
    return TestProblem(obj_func=obj_func, search_space=DUMMY_SEARCH_SPACE)


# --- get_optimizer_overhead tests ---


def test_get_optimizer_overhead():
    """get_optimizer_overhead should return sampling time data with correct structure."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, expensive_sampler=False, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, n_trials=n_trials)

    overhead = get_overhead_from_study(study)
    assert "before_sample" in overhead
    assert "after_sample" in overhead
    assert "worker_index" in overhead
    # n_trials + n_workers - 1 ask() calls
    expected_len = n_trials + n_workers - 1
    assert len(overhead["before_sample"]) == expected_len
    assert len(overhead["after_sample"]) == expected_len
    assert len(overhead["worker_index"]) == expected_len
    # after_sample >= before_sample
    for before, after in zip(overhead["before_sample"], overhead["after_sample"]):
        assert after >= before


# --- property tests ---


def test_simulator_properties():
    """Public properties should return expected values."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, expensive_sampler=False, allow_parallel_sampling=False)
    assert simulator._n_workers == n_workers


# --- result ordering with expensive_sampler=True ---


def test_results_sorted_by_cumtime_with_expensive_sampler():
    """With expensive_sampler=True, results should have cumtimes in non-decreasing order."""
    n_trials = 8
    simulator = AsyncOptBenchmarkSimulator(n_workers=4, expensive_sampler=True, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem(obj_func=simplest_dummy_func)
    simulator.optimize(study, problem, n_trials=n_trials)

    results = get_results_from_study(study)
    cumtimes = np.array(results["cumtime"])
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


def test_results_cumtime_monotonic_without_expensive_sampler():
    """Without expensive_sampler, results cumtimes should also be non-decreasing."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, expensive_sampler=False, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, n_trials=n_trials)

    results = get_results_from_study(study)
    cumtimes = np.array(results["cumtime"])
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


# --- tell_pending_result edge cases ---


def test_tell_skips_none_pending_results():
    """_tell_pending_result should skip workers with None pending results."""
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=2, expensive_sampler=False, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, n_trials=n_trials)

    results = get_results_from_study(study)
    assert len(results["cumtime"]) == n_trials


def test_multi_worker_all_results_collected():
    """With multiple workers, all n_trials results should be collected."""
    n_workers = 4
    n_trials = 12
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, expensive_sampler=False, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, n_trials=n_trials)

    results = get_results_from_study(study)
    assert len(results["cumtime"]) == n_trials
    assert len(results["objectives"]) == n_trials
    assert len(results["worker_index"]) == n_trials


if __name__ == "__main__":
    unittest.main()
