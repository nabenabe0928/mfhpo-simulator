from __future__ import annotations

import json
import multiprocessing
import os
import time
from argparse import ArgumentParser

from benchmark_apis import MFHartmann

from benchmark_simulator import ObjectiveFuncType, ObjectiveFuncWrapper

import ConfigSpace as CS

import numpy as np


parser = ArgumentParser()
parser.add_argument("--n_evals", type=int, default=100)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--runtime_factor", type=float, default=100.0)
parser.add_argument("--raw", type=str, choices=["True", "False"], default="False")
parser.add_argument("--deterministic", type=str, choices=["True", "False"], default="False")
parser.add_argument("--mode", type=str, choices=["single", "multi", "no"], default="single")
args = parser.parse_args()

FIDEL_KEY = "z0"
OBJ_KEY = "loss"
RUNTIME_KEY = "runtime"
RAW = eval(args.raw)
MODE = args.mode
DETERMINISTIC = eval(args.deterministic)
N_WORKERS = args.n_workers
N_EVALS = args.n_evals
SEED = args.seed
RUNTIME_FACTOR = args.runtime_factor


class RandomOptimizer:
    def __init__(
        self,
        func: ObjectiveFuncType,
        n_workers: int,
        n_evals: int,
        config_space: CS.ConfigurationSpace,
        fidel_range: tuple[int, int],
        seed: int | None,
    ):
        self._func = func
        self._n_workers = n_workers
        rng = np.random.RandomState(seed)
        config_space.seed(seed)
        self._configs = [config.get_dictionary() for config in config_space.sample_configuration(n_evals)]
        self._fidels = rng.randint(low=fidel_range[0], high=fidel_range[1], size=n_evals)
        self._index = 0
        self.timestamps = []

    def ask(self):
        config, fidel = self._configs[self._index], self._fidels[self._index]
        self._index += 1
        return config, {FIDEL_KEY: fidel}, None

    def tell(self, *args, **kwargs):
        self.timestamps.append(time.time())

    def optimize(self) -> list[dict[str, float]]:
        pool = multiprocessing.Pool(processes=self._n_workers)

        _results = []
        for config, fidel in zip(self._configs, self._fidels):
            _results.append(pool.apply_async(self._func, kwds=dict(eval_config=config, fidel=fidel)))

        pool.close()
        pool.join()
        return [r.get() for r in _results]


class MyObjectiveFuncWrapper(ObjectiveFuncWrapper):
    def __call__(self, eval_config: dict[str, float], fidel: int) -> dict[str, float]:
        results = super().__call__(eval_config=eval_config, fidels=dict(z0=fidel))
        return dict(loss=results[self.obj_keys[0]], timestamp=time.time())


class MyObjectiveFuncSleep:
    def __init__(self, func: MFHartmann):
        self._func = func

    def __call__(self, eval_config: dict[str, float], fidel: int) -> dict[str, float]:
        results = self._func(eval_config=eval_config, fidels=dict(z0=fidel))
        time.sleep(results[RUNTIME_KEY])
        return dict(loss=results[OBJ_KEY], timestamp=time.time())


def extract(results: dict, start: float) -> tuple[np.ndarray, np.ndarray]:
    loss_vals = np.array([r["loss"] for r in results])
    timestamp = np.array([r["timestamp"] for r in results])
    order = np.argsort(timestamp)[:N_EVALS]
    return loss_vals[order], timestamp[order] - start


def run_with_wrapper(
    bench: MFHartmann, seed: int, ask_and_tell: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapper = MyObjectiveFuncWrapper(
        obj_func=bench,
        fidel_keys=["z0"],
        n_workers=N_WORKERS,
        n_actual_evals_in_opt=N_EVALS + N_WORKERS,
        n_evals=N_EVALS,
        ask_and_tell=ask_and_tell,
        careful_init=True,
    )
    opt = RandomOptimizer(
        func=wrapper,
        n_workers=N_WORKERS,
        n_evals=N_EVALS + N_WORKERS,
        config_space=bench.config_space,
        fidel_range=(bench.min_fidels[FIDEL_KEY], bench.max_fidels[FIDEL_KEY]),
        seed=seed,
    )
    start = time.time()
    if ask_and_tell:
        wrapper.simulate(opt)
        results = [dict(timestamp=timestamp, loss=0.0) for timestamp in opt.timestamps]
    else:
        results = opt.optimize()

    _, actual_cumtime = extract(results, start=start)
    results = wrapper.get_results()

    return np.array(results["loss"][:N_EVALS]), np.array(results["cumtime"][:N_EVALS]), actual_cumtime


def run_without_wrapper(bench: MFHartmann, seed: int) -> tuple[np.ndarray, np.ndarray]:
    wrapper = MyObjectiveFuncSleep(func=bench)
    start = time.time()
    results = RandomOptimizer(
        func=wrapper,
        n_workers=N_WORKERS,
        n_evals=N_EVALS + N_WORKERS,
        config_space=bench.config_space,
        fidel_range=(bench.min_fidels[FIDEL_KEY], bench.max_fidels[FIDEL_KEY]),
        seed=seed,
    ).optimize()
    return extract(results, start=start)


def get_result_without_simulator(bench, seed: int) -> tuple[list[float], list[float], None]:
    loss_vals, actual_cumtime = run_without_wrapper(bench, seed=seed)
    return loss_vals.tolist(), actual_cumtime.tolist(), None


def get_result_with_single_core_simulator(bench, seed: int) -> tuple[list[float], list[float], list[float]]:
    loss_vals, simulated_cumtime, actual_cumtime = run_with_wrapper(bench, seed=seed, ask_and_tell=True)
    return loss_vals.tolist(), actual_cumtime.tolist(), simulated_cumtime.tolist()


def get_result_with_multi_core_simulator(bench, seed: int) -> tuple[list[float], list[float], list[float]]:
    loss_vals, simulated_cumtime, actual_cumtime = run_with_wrapper(bench, seed=seed)
    return loss_vals.tolist(), actual_cumtime.tolist(), simulated_cumtime.tolist()


def main(deterministic: bool):
    data = {k: dict(loss=[], actual_cumtime=[], simulated_cumtime=[]) for k in ["naive", "ours", "ours_ask_and_tell"]}
    bench = MFHartmann(dim=6, runtime_factor=RUNTIME_FACTOR, deterministic=deterministic)
    for seed in range(SEED):
        print(f"Run with {seed=}")

        for key, result_fn in {
            "ours": get_result_with_multi_core_simulator,
            "ours_ask_and_tell": get_result_with_single_core_simulator,
            "naive": get_result_without_simulator,
        }.items():
            bench.reseed(seed)
            loss_vals, actual_cumtime, simulated_cumtime = result_fn(bench, seed=seed)
            data[key]["loss"].append(loss_vals)
            data[key]["actual_cumtime"].append(actual_cumtime)
            data[key]["simulated_cumtime"].append(simulated_cumtime)

        suffix = "deterministic" if deterministic else "noisy"
        with open(f"demo/validation-results-{suffix}.json", mode="w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    if not RAW:
        main(deterministic=True)
        main(deterministic=False)
    else:
        bench = MFHartmann(dim=6, runtime_factor=RUNTIME_FACTOR, deterministic=DETERMINISTIC)

        if MODE == "single":
            loss, actual_cumtime, simulated_cumtime = get_result_with_single_core_simulator(bench, SEED)
        elif MODE == "multi":
            loss, actual_cumtime, simulated_cumtime = get_result_with_multi_core_simulator(bench, SEED)
        else:
            loss, actual_cumtime = get_result_without_simulator(bench, SEED)

        dir_name = f"demo/random_{MODE}_noise={not DETERMINISTIC}"
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, f"{SEED:0>3}.json"), mode="w") as f:
            json.dump(dict(loss=loss, actual_cumtime=actual_cumtime, simulated_cumtime=simulated_cumtime), f, indent=4)
