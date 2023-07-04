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

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None]:
        # Ask method is mandatory
        trial = self._study.ask(self._distributions)
        return trial.params, None, trial._trial_id

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
        self._study.tell(trial=config_id, values=results[self._obj_key])


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
