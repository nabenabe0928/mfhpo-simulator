# A Simulator for Multi-Fidelity or Parallel Optimization Using Tabular or Surrogate Benchmarks

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

## Motivation

When we run parallel optimization experiments using tabular or surrogate benchmarks, each evaluation must be ordered based on the runtime that each configuration, in reality, takes.
However, the evaluation of tabular or surrogate benchmarks, by design, does not take long.
For this reason, the timing of each configuration taken into account must be ordered as if we evaluated each configuration.

In this package, we automatically sort out this problem by pending to pass the hyperparameter configurations to be evaluated internally, and in turn, we obtain the right order of each hyperparameter configuration to be evaluated.

**NOTE**

Our wrapper assumes that none of the workers will not die and any additional workers will not be added after the initialization.
Therefore, if any workers die, our current wrapper hangs and keeps warning except we provide `max_waiting_time` for the instantiation.
I am not sure if I will support any additional workers after the initialization yet.
Furthermore, our package cannot be run on Windows OS because the Python module `fcntl` is not supported on Windows OS.

## Setup

The installation is easily done by `pip-install`:

```shell
$ pip install mfhpo-simulator
```

The requirements are:
- Unix system
- Python 3.8 or later

The dependencies of this package are only **numpy** and **ujson**. 

## Test

The very minimal example is provided in [examples/minimal.py](examples/minimal.py) and you can run by `python -m examples.minimal` after the setup below.

If you would like to know the examples of the usage, please clone the whole repository:

```shell
$ git clone mfhpo-simulator
$ pip install -r requirements-for-developer.txt
```

Note that the environment for SMAC3 is separately defined due to some dependency issues:

```shell
$ pip install -r requirements-for-smac.txt
```

Additionally, we need to increase the `retries` in the following parts:
- intensifier/intensifier.py (L53)
- facade/abstract_facade.py (L414)
- main/config_selector.py (L54)
For example, we used `retries=100` to prevent unnecessary termination of executions.

Then you can run various examples:

```shell
# Run BOHB with 4 workers on dataset_id=0 in HPOlib (HPOlib has 4 different datasets)
$ python -m examples.bohb --seed 0 --dataset_id 0 --bench_name hpolib --n_workers 4

# Run DEHB with 4 workers on dataset_id=0 in LCBench (LCBench has 34 different datasets)
$ python -m examples.dehb --seed 0 --dataset_id 0 --bench_name lc --n_workers 4

# Run NePS with 4 workers on dataset_id=0 in JAHS-Bench-201 (JAHS-Bench-201 has 3 different datasets)
$ ./examples/neps.sh --seed 0 --dataset_id 0 --bench_name jahs --n_workers 4

# Run SMAC3 with 4 workers on Hartmann (Hartmann has 3 or 6 dimensions)
$ python -m examples.smac --seed 0 --dim 3 --bench_name hartmann --n_workers 4
```

Each argument is defined as follows:
1. `--seed` (`int`): The random seed to be used.
2. `--bench_name` (`Literal["hpolib", "jahs", "lc", "hartmann", "branin"]`): The benchmark to be used.
3. `--n_workers` (`int`): The number of parallel workers to be used. Too high numbers may crash your system because the specified benchmark dataset must stay on the memory for each process/thread.
4. `--dataset_id` (`Optional[int]`): The dataset ID to be used in the specified dataset (0 to 3 for HPOlib, 0 to 33 for LCBench, and 0 t0 2 for JAHS-Bench-201). The default value is 0.
5. `--dim` (`Optional[int]`): The dimensionality of the Hartmann function and it is used only Hartmann function. The default value is 3.

Note that `--seed` does not guarantee the reproducitility because of the parallel processing nature.

## Arguments of ObjectiveFuncWorker/CentralWorkerManager

In most packages, users need to use `CentralWorkerManager`.
However, [`BOHB`](https://github.com/automl/hpBandSter/) and [`NePS`](https://github.com/automl/neps) are exceptions where you need to instantiate `ObjectiveFuncWorker`.
Basically, we need to use `ObjectiveFuncWorker` for BOHB and NePS because they share the information in each worker via some types of server or they launch multiple independent threads.
On the other hand, when optimizers use typical multiprocessing/multithreading packages such as `multiprocessing`, `threading`, `concurrent.futures`, `joblib`, `dask`, and `mpi4py`, users need to use `CentralWorkerManager`.
Both `ObjectiveFuncWorker` and `CentralWorkerManager` share the same user interface and I describe each argument of both classes here:
1. `subdir_name` (`str`): The directory to store the information,
2. `n_workers` (`int`): The number of parallel workers,
3. `obj_func` (`ObjectiveFuncType`): The objective function to be wrapped. See [`ObjectiveFuncType`](https://github.com/nabenabe0928/mfhpo-simulator/blob/main/benchmark_simulator/_constants.py#L10-L43) for more details,
4. `n_actual_evals_in_opt` (`int`): The number of evaluations inside the optimiziers (this argument will be used only for raising an error),
5. `n_evals` (`int`): The number of evaluations to be stored in the information,
6. `continual_max_fidel` (`Optional[int]`): The maximum fidelity value used for the continual evaluation (it is valid only if we have a single fidelity). If `None`, we just do a normal asynchronous or multi-fidelity optimization. Note that continual evaluation is to train each hyperparameter configuration from scratch or from intermediate results. For example, when we have a train result of a neural network with a hyperparameter configuration `A` for 10 epochs, we train a neural network with `A` for 30 epochs from 10 epochs rather than from scratch,
7. `obj_keys` (`List[str]`): The list of objective names in the output from `obj_func`,
8. `runtime_key` (`str`): The key is for runtime. The output of objective function must include runtime,
9. `obj_keys` (`List[str]`): The list of fidelity names that will be feeded to the objective function,
10. `seed` (`Optional[int]`): The random seed to be used in each worker,
11. `max_waiting_time` (`float`): The maximum waiting time for each worker. If workers wait for the provided amount of time, the wrapper will return only `INF`, and
12. `store_config` (`bool`): Whether to store configuration, fidelities, and seed for each evaluation. It consumes much more storage when you use it for large-scale experiments.

## Citation

Please use the following format for the citation of this repository:

```
@inproceedings{watanabe2023mfo-simulator,
  title   = {{P}ython Wrapper for Simulating Multi-Fidelity Optimization on HPO Benchmarks without Any Wait},
  author  = {S. Watanabe},
  year    = {2023},
  journal = {arXiv:XXXX.XXXXX}
}
```
