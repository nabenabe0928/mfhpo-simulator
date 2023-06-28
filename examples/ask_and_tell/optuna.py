from __future__ import annotations

from typing import Any

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import ConfigSpace as CS

from examples.utils import get_bench_instance, get_save_dir_name, parse_args

import optuna


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


class OptunaWrapper(AbstractAskTellOptimizer):
    """
    NOTE:
        We found out that it is hard to simulate multi-fidelity in Optuna,
        because Optuna does not take fidelity parameters as an argument and it internally processes them.
        However, if we do not use Pruner, which is for multi-fidelity optimization, it works fine.
    """

    def __init__(self, distributions: dict[str, optuna.distributions.BaseDistribution], obj_key: str):
        self._study = optuna.create_study()
        self._obj_key = obj_key
        self._distributions = distributions
        self._hp_names = list(self._distributions.keys())
        self._pending_trials: dict[int, list[optuna.Trial]] = {}

    ######################################
    # ask and tell methods are mandatory #
    ######################################

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        # Ask method is mandatory
        trial = self._study.ask(self._distributions)
        self._store_trial(trial)
        return trial.params, None

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
    ) -> None:
        # Tell method is mandatory
        trial = self._fetch_trial(eval_config)
        self._study.tell(trial, results[self._obj_key])

    #######################################################
    # Custom methods due to the Optuna's algorithm design #
    #######################################################

    def _calculate_config_hash(self, eval_config: dict[str, Any]) -> int:
        # It may not be the safest option, but it is sufficient for this example.
        return int(hash(str({k: eval_config[k] for k in self._hp_names})))

    def _store_trial(self, trial: optuna.Trial) -> None:
        config_hash = self._calculate_config_hash(trial.params)
        if config_hash in self._pending_trials:
            self._pending_trials[config_hash].append(trial)
        else:
            self._pending_trials[config_hash] = [trial]

    def _fetch_trial(self, eval_config: dict[str, Any]) -> optuna.Trial:
        config_hash = self._calculate_config_hash(eval_config)
        trial = self._pending_trials[config_hash].pop(0)
        if len(self._pending_trials[config_hash]) == 0:
            # Empty the cache if the length is zero.
            self._pending_trials.pop(config_hash)

        return trial


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    bench = get_bench_instance(args, keep_benchdata=True)
    opt = OptunaWrapper(distributions=get_distributions(bench.config_space), obj_key="loss")
    worker = ObjectiveFuncWrapper(
        save_dir_name=save_dir_name,
        ask_and_tell=True,
        n_workers=args.n_workers,
        obj_func=bench,
        n_actual_evals_in_opt=105,
        n_evals=100,
        seed=args.seed,
    )
    worker.simulate(opt)
