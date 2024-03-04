from __future__ import annotations

from typing import Any

from benchmark_simulator import ObjectiveFuncWrapper

import optuna


def objective(eval_config: dict[str, float], **kwargs) -> dict[str, float]:
    x0 = eval_config["x0"]
    x1 = eval_config["x1"]
    f0 = x0**2 + x1**2
    f1 = (x0 - 2) ** 2 + (x1 - 2) ** 2
    runtime = 50 - f0
    return {"f0": f0, "f1": f1, "constraint": x0 + x1, "runtime": runtime}


def constraint(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    constraint_key = "constraint"
    return (trial.user_attrs[constraint_key],)


class OptunaObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # 0. Adapt the callable of the objective function to the Optuna interface at https://github.com/optuna/optuna/
    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        eval_config = {
            "x0": trial.suggest_float("x0", low=-5.0, high=5.0),
            "x1": trial.suggest_float("x1", low=-5.0, high=5.0),
        }
        results = super().__call__(eval_config=eval_config)
        constraint_key = "constraint"
        assert self.obj_keys[2] == constraint_key
        trial.set_user_attr(constraint_key, results[constraint_key])
        assert self.obj_keys[0] == "f0" and self.obj_keys[1] == "f1"
        return [results[self.obj_keys[0]], results[self.obj_keys[1]]]


if __name__ == "__main__":
    # 1. Define a wrapper instance (Default is n_workers=4, but you can change it from the argument)
    wrapper = OptunaObjectiveFuncWrapper(obj_func=objective, obj_keys=["f0", "f1", "constraint"])
    sampler = optuna.samplers.TPESampler(constraints_func=constraint)
    study = optuna.create_study(sampler=sampler, directions=["minimize"] * 2)

    # 2. Feed the wrapped objective function to the optimizer directly
    # 3. just start the optimization!
    study.optimize(wrapper, n_trials=wrapper.n_actual_evals_in_opt, n_jobs=wrapper.n_workers)
