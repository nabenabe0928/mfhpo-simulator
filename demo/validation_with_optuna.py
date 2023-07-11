from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from typing import Any

from benchmark_apis import MFHartmann

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import ConfigSpace as CS

import numpy as np

import optuna


parser = ArgumentParser()
parser.add_argument("--n_evals", type=int, default=100)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--n_seeds", type=int, default=10)
parser.add_argument("--runtime_factor", type=float, default=100.0)
args = parser.parse_args()

FIDEL_KEY = "z0"
OBJ_KEY = "loss"
RUNTIME_KEY = "runtime"
N_WORKERS = args.n_workers
N_EVALS = args.n_evals
N_SEEDS = args.n_seeds
RUNTIME_FACTOR = args.runtime_factor


def get_distributions(config_space: CS.ConfigurationSpace) -> dict[str, optuna.distributions.BaseDistribution]:
    dists: dict[str, optuna.distributions.BaseDistribution] = {}
    for name in config_space:
        hp = config_space.get_hyperparameter(name)
        if hasattr(hp, "choices"):
            dists[name] = optuna.distributions.CategoricalDistribution(choices=hp.choices)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            dists[name] = optuna.distributions.FloatDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            dists[name] = optuna.distributions.IntDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        else:
            raise ValueError(f"{type(hp)} is not supported in get_distributions.")

    return dists


def get_config_from_trial(
    trial: optuna.Trial, config_space: CS.ConfigurationSpace
) -> tuple[dict[str, int | float | str | bool], dict[str, int | float]]:
    eval_config: dict[str, int | float | str | bool] = {}
    fidels: dict[str, int | float] = {}
    for name in config_space:
        hp = config_space.get_hyperparameter(name)
        if hasattr(hp, "choices"):
            val = trial.suggest_categorical(name=name, choices=hp.choices)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            val = trial.suggest_float(name=name, low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            val = trial.suggest_int(name=name, low=hp.lower, high=hp.upper, log=hp.log)
        else:
            raise ValueError(f"{type(hp)} is not supported in get_distributions.")

        if name == FIDEL_KEY:
            fidels[name] = val
        else:
            eval_config[name] = val

    return eval_config, fidels


class OptunaAskTellOptimizer(AbstractAskTellOptimizer):
    def __init__(
        self, distributions: dict[str, optuna.distributions.BaseDistribution], sampler: optuna.samplers.TPESampler
    ):
        self._study = optuna.create_study(sampler=sampler)
        self._distributions = distributions

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        # Ask method is mandatory
        trial = self._study.ask(self._distributions)
        eval_config = trial.params.copy()
        fidels = {FIDEL_KEY: eval_config.pop(FIDEL_KEY)}
        return eval_config, fidels, trial._trial_id

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
        config_id: int | None,
    ) -> None:
        assert isinstance(config_id, int)  # mypy redefinition
        # Tell method is mandatory
        self._study.tell(trial=config_id, values=results[OBJ_KEY])


class OptunaObjectiveFuncWrapper(ObjectiveFuncWrapper):
    def __init__(self, config_space: CS.ConfigurationSpace, **kwargs):
        super().__init__(**kwargs)
        self._config_space = config_space

    def __call__(self, trial: optuna.Trial) -> float:
        eval_config, fidels = get_config_from_trial(trial=trial, config_space=self._config_space)
        results = super().__call__(eval_config=eval_config, fidels=fidels)
        return results[OBJ_KEY]


class OptunaObjectiveSleepFuncWrapper:
    def __init__(self, func: MFHartmann, config_space: CS.ConfigurationSpace):
        self._func = func
        self._config_space = config_space

    def __call__(self, trial: optuna.Trial) -> float:
        eval_config, fidels = get_config_from_trial(trial=trial, config_space=self._config_space)
        results = self._func(eval_config=eval_config, fidels=fidels)
        time.sleep(results[RUNTIME_KEY])
        return results[OBJ_KEY]


def extract(study: optuna.Study) -> tuple[np.ndarray, np.ndarray]:
    start = study.trials[0].datetime_start.timestamp()
    actual_cumtimes = np.array(
        [t.datetime_complete.timestamp() - start for t in study.trials if t.datetime_complete is not None]
    )
    loss_vals = np.array([t.values[0] for t in study.trials if t.values is not None])
    order = np.argsort(actual_cumtimes)[:N_EVALS]
    return loss_vals[order], actual_cumtimes[order]


def run_with_wrapper(
    bench: MFHartmann, seed: int, config_space: CS.ConfigurationSpace, ask_and_tell: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapper = OptunaObjectiveFuncWrapper(
        config_space=config_space,
        obj_func=bench,
        fidel_keys=["z0"],
        n_workers=N_WORKERS,
        n_actual_evals_in_opt=N_EVALS + N_WORKERS,
        n_evals=N_EVALS,
        ask_and_tell=ask_and_tell,
        careful_init=True,
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    if ask_and_tell:
        opt = OptunaAskTellOptimizer(distributions=get_distributions(config_space), sampler=sampler)
        wrapper.simulate(opt)
        study = opt._study
    else:
        study = optuna.create_study(sampler=sampler)
        study.optimize(wrapper, n_trials=wrapper.n_actual_evals_in_opt, n_jobs=wrapper.n_workers)

    loss_vals, actual_cumtime = extract(study)
    results = wrapper.get_results()
    assert np.allclose(results["loss"][:N_EVALS], loss_vals)  # sanity check
    return loss_vals, np.array(results["cumtime"][:N_EVALS]), actual_cumtime


def run_without_wrapper(
    bench: MFHartmann, seed: int, config_space: CS.ConfigurationSpace
) -> tuple[np.ndarray, np.ndarray]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    wrapper = OptunaObjectiveSleepFuncWrapper(func=bench, config_space=config_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(wrapper, n_trials=N_EVALS, n_jobs=N_WORKERS)

    start = study.trials[0].datetime_start.timestamp()
    actual_cumtimes = np.array([t.datetime_complete.timestamp() - start for t in study.trials])
    loss_vals = np.array([t.values[0] for t in study.trials])
    order = np.argsort(actual_cumtimes)
    return loss_vals[order], actual_cumtimes[order]


def main(deterministic: bool):
    data = {
        "naive": {
            "loss": [],
            "actual_cumtime": [],
        },
        "ours": {
            "loss": [],
            "actual_cumtime": [],
            "simulated_cumtime": [],
        },
        "ours_ask_and_tell": {
            "loss": [],
            "actual_cumtime": [],
            "simulated_cumtime": [],
        },
    }
    bench = MFHartmann(dim=6, runtime_factor=RUNTIME_FACTOR, deterministic=deterministic)
    config_space = bench.config_space
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            name=FIDEL_KEY,
            lower=bench.min_fidels[FIDEL_KEY],
            upper=bench.max_fidels[FIDEL_KEY],
        )
    )
    for seed in range(N_SEEDS):
        print(f"Run with {seed=}")

        time.sleep(0.01)
        bench.reseed(seed)
        loss_vals, simulated_cumtime, actual_cumtime = np.array(
            run_with_wrapper(bench, seed=seed, config_space=config_space)
        )
        data["ours"]["loss"].append(loss_vals.tolist())
        data["ours"]["actual_cumtime"].append(actual_cumtime.tolist())
        data["ours"]["simulated_cumtime"].append(simulated_cumtime.tolist())

        time.sleep(0.01)
        bench.reseed(seed)
        loss_vals, simulated_cumtime, actual_cumtime = np.array(
            run_with_wrapper(bench, seed=seed, ask_and_tell=True, config_space=config_space)
        )
        data["ours_ask_and_tell"]["loss"].append(loss_vals.tolist())
        data["ours_ask_and_tell"]["actual_cumtime"].append(actual_cumtime.tolist())
        data["ours_ask_and_tell"]["simulated_cumtime"].append(simulated_cumtime.tolist())

        bench.reseed(seed)
        loss_vals, actual_cumtime = np.array(run_without_wrapper(bench, seed=seed, config_space=config_space))
        data["naive"]["loss"].append(loss_vals.tolist())
        data["naive"]["actual_cumtime"].append(actual_cumtime.tolist())

        naive_loss, our_loss = data["naive"]["loss"][-1], data["ours"]["loss"][-1]
        percent = 100 * np.sum(np.isclose(naive_loss, our_loss)) / len(our_loss)
        print(f"How much was correct?: {percent:.2f}%. NOTE: Tie-break could cause False!")

        suffix = "deterministic" if deterministic else "noisy"
        with open(f"demo/validation-optuna-results-{suffix}.json", mode="w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main(deterministic=True)
    main(deterministic=False)
