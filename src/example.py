from __future__ import annotations

from typing import TYPE_CHECKING

import optunahub

import optuna


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


# optunahub.load_module("benchmarks/async_opt_simulator").AsyncOptBenchmarkSimulator
AsyncOptBenchmarkSimulator = optunahub.load_local_module("src", registry_root="./").AsyncOptBenchmarkSimulator
sim = AsyncOptBenchmarkSimulator(n_workers=4)


class WrappedProblem(optunahub.load_module("benchmarks/hpolib").Problem):
    def __init__(self, *, metric_names: list[str], **kwargs: Any) -> None:
        runtime_key = "train_time"
        self._is_runtime_in_objective = runtime_key in metric_names
        metric_names = [name for name in metric_names if name != runtime_key] + [runtime_key]
        super().__init__(metric_names=metric_names, **kwargs)

    def __call__(self, trial: optuna.Trial) -> float | Sequence[float]:
        output = super().__call__(trial)
        trial.set_user_attr("runtime", output[-1])
        if self._is_runtime_in_objective:
            return output
        return output[:-1]

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        directions = super().directions
        if self._is_runtime_in_objective:
            return directions
        return directions[:-1]


problem = WrappedProblem(dataset_id=0, metric_names=["val_loss"])
study = optuna.create_study(directions=problem.directions)
sim.optimize(study, problem, n_trials=100)
print(sim.get_results_from_study(study))
