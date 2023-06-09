from __future__ import annotations

from typing import Any

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

import ConfigSpace as CS

from examples.utils import get_bench_instance, get_save_dir_name, parse_args


class RandomOptimizer:
    def __init__(self, config_space: CS.ConfigurationSpace, max_fidels: dict[str, int | float]):
        self._config_space = config_space
        self._max_fidels = max_fidels

    def ask(self) -> dict[str, Any]:
        return self._config_space.sample_configuration().get_dictionary()


class RandomOptimizerWrapper(AbstractAskTellOptimizer):
    def __init__(self, opt: RandomOptimizer):
        self._opt = opt

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None, int | None]:
        eval_config = self._opt.ask()
        return eval_config, self._opt._max_fidels, None

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None,
        config_id: int | None,
    ) -> None:
        pass


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    bench = get_bench_instance(args, keep_benchdata=True)
    opt = RandomOptimizerWrapper(RandomOptimizer(bench.config_space, bench.max_fidels))
    worker = ObjectiveFuncWrapper(
        save_dir_name=save_dir_name,
        ask_and_tell=True,
        n_workers=args.n_workers,
        obj_func=bench,
        n_actual_evals_in_opt=105,
        n_evals=100,
        seed=args.seed,
        fidel_keys=bench.fidel_keys,
    )
    worker.simulate(opt)
