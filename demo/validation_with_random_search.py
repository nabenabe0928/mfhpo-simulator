from __future__ import annotations

import json
import multiprocessing
import time
from argparse import ArgumentParser

from benchmark_apis import MFHartmann

from benchmark_simulator import ObjectiveFuncType, ObjectiveFuncWrapper

import ConfigSpace as CS

import numpy as np


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

    def optimize(self) -> list[float]:
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
    for seed in range(N_SEEDS):
        print(f"Run with {seed=}")

        time.sleep(0.01)
        bench.reseed(seed)
        loss_vals, simulated_cumtime, actual_cumtime = np.array(run_with_wrapper(bench, seed=seed))
        data["ours"]["loss"].append(loss_vals.tolist())
        data["ours"]["actual_cumtime"].append(actual_cumtime.tolist())
        data["ours"]["simulated_cumtime"].append(simulated_cumtime.tolist())

        time.sleep(0.01)
        bench.reseed(seed)
        loss_vals, simulated_cumtime, actual_cumtime = np.array(run_with_wrapper(bench, seed=seed, ask_and_tell=True))
        data["ours_ask_and_tell"]["loss"].append(loss_vals.tolist())
        data["ours_ask_and_tell"]["actual_cumtime"].append(actual_cumtime.tolist())
        data["ours_ask_and_tell"]["simulated_cumtime"].append(simulated_cumtime.tolist())

        bench.reseed(seed)
        loss_vals, actual_cumtime = np.array(run_without_wrapper(bench, seed=seed))
        data["naive"]["loss"].append(loss_vals.tolist())
        data["naive"]["actual_cumtime"].append(actual_cumtime.tolist())

        naive_loss, our_loss = data["naive"]["loss"][-1], data["ours"]["loss"][-1]
        percent = 100 * np.sum(np.isclose(naive_loss, our_loss)) / len(our_loss)
        print(f"How much was correct?: {percent:.2f}%. NOTE: Tie-break could cause False!")

        suffix = "deterministic" if deterministic else "noisy"
        with open(f"demo/validation-results-{suffix}.json", mode="w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main(deterministic=True)
    main(deterministic=False)
