# A Simulator for Multi-Fidelity or Parallel Optimization Using Tabular or Surrogate Benchmarks

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

## Motivation

<table>
    <tr>
        <td><img src="figs/compress-conceptual.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="figs/api-conceptual.png" alt=""></td>
    </tr>
</table>

When we run parallel optimization experiments using tabular or surrogate benchmarks, each evaluation must be ordered based on the runtime that each configuration, in reality, takes.
However, the evaluation of tabular or surrogate benchmarks, by design, does not take long.
For this reason, the timing of each configuration taken into account must be ordered as if we evaluated each configuration.

In this package, we automatically sort out this problem by pending to pass the hyperparameter configurations to be evaluated internally, and in turn, we obtain the right order of each hyperparameter configuration to be evaluated.
If the optimizer interface is the [ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) interface, users can pass the optimizer to a simulator directly and the simulator automatically performs the optimization loop as if function calls are run in parallel.

| Arguments | What Wrapper | Function Call | Requirements | Benefits | Downsides |
|--|:--:|:--:|--|--|--|
| Default | Function  | Parallel | Optimizer does intra-process synchronization (e.g. [DEHB](examples/dehb.py) and [SMAC3](examples/smac.py)) | No need to change the optimizer interface and reproduce exactly how optimizers run | Could be very slow, unstable, and memory-intensive with a large `n_workers` |
|`launch_multiple_wrappers_from_user_side=True`  | Function  | Parallel | Optimizer does inter-process synchronization (e.g. [NePS](examples/neps.py) and [BOHB](examples/bohb.py)) | Same above | Same above |
|`ask_and_tell=True` | Function and Optimizer | *Sequential | Optimizer must take the [ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) interface (see [example](examples/ask_and_tell/)) | Fast, stable, and memory-efficient even with a large `n_workers` | Force the ask-and-tell interface and may unexpectedly ignore the memory bottleneck that could be caused by parallel runs |

\* It runs function call sequentially, but function calls are internally processed as if they are run in parallel.

While users do not have to change the interface of optimizers for `ask_and_tell=False` and only need to change the interface of objective function, users may need to change the interface of both for `ask_and_tell=True`.
In principle, `ask_and_tell=True` requires optimizers to be the ask-and-tell interface.
In exchange for the strict constraint, it stabilizes the simulation.

**NOTE**

Our wrapper assumes that none of the workers will not die and any additional workers will not be added after the initialization.
Therefore, if any workers die, our current wrapper hangs and keeps warning except we provide `max_waiting_time` for the instantiation.
We are not sure if we will support any additional workers after the initialization yet.
Furthermore, our package **cannot be run on Windows OS** because the Python module `fcntl` is not supported on Windows OS.
Although our package supports MacOS, it is advisable to use Linux system.

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

The very minimal examples are provided in [examples/minimal/](examples/minimal/) and you can run these examples by:
```shell
# DEHB
$ python -m examples.minimal.dehb

# BOHB
$ python -m examples.minimal.bohb

# SMAC
$ python -m examples.minimal.smac

# NePS
$ ./examples/minimal/neps.sh --n_workers 4
```

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

## Arguments for Function Wrapper

In most packages, users need to use the default setting (`launch_multiple_wrappers_from_user_side=False` and `ask_and_tell=True`).
However, [`BOHB`](https://github.com/automl/hpBandSter/) and [`NePS`](https://github.com/automl/neps/) are exceptions where you need to use `launch_multiple_wrappers_from_user_side=True`.
Basically, we need to use `launch_multiple_wrappers_from_user_side=True` for BOHB and NePS because they explicitly instantiate multiple objective function objects from the user side and each of them has its own main process.
On the other hand, when optimizers use typical multiprocessing/multithreading packages such as `multiprocessing`, `threading`, `concurrent.futures`, `joblib`, `dask`, and `mpi4py`, the main process can easily communicate with each child process, and hence users can stick to the default setting.
Each argument of `ObjectiveFuncWrapper` is the following:
1. `obj_func` (`ObjectiveFuncType`): The objective function to be wrapped. See [`ObjectiveFuncType`](https://github.com/nabenabe0928/mfhpo-simulator/blob/main/benchmark_simulator/_constants.py#L40-L73) for more details,
2. `launch_multiple_wrappers_from_user_side` (`bool`): Whether users need to launch multiple objective function wrappers from user side,
3. `ask_and_tell` (`bool`): Whether to use an ask-and-tell interface optimizer and simulate the optimization in `ObjectiveFuncWrapper`,
4. `save_dir_name` (`str | None`): The directory to store the information and it must be specified when using `launch_multiple_wrappers_from_user_side=True`, otherwise the directory name will be automatically generated,
5. `n_workers` (`int`): The number of parallel workers,
6. `n_actual_evals_in_opt` (`int`): The number of evaluations inside the optimizers (this argument will be used only for raising an error),
7. `n_evals` (`int`): The number of evaluations to be stored in the information,
8. `continual_max_fidel` (`int | None`): The maximum fidelity value used for the continual evaluation (it is valid only if we have a single fidelity). If `None`, we just do a normal asynchronous or multi-fidelity optimization. Note that continual evaluation is to train each hyperparameter configuration from scratch or from intermediate results. For example, when we have a training result of a neural network with a hyperparameter configuration `A` for 10 epochs, we train a neural network with `A` for 30 epochs from 10 epochs rather than from scratch,
9. `obj_keys` (`list[str]`): The list of objective names in the output from `obj_func`,
10. `runtime_key` (`str`): The key is for runtime. The output of the objective function must include runtime,
11. `obj_keys` (`list[str]`): The list of fidelity names that will be fed to the objective function,
12. `seed` (`int | None`): The random seed to be used in each worker,
13. `max_waiting_time` (`float`): The maximum waiting time for each worker. If workers wait for the provided amount of time, the wrapper will return only `INF`, and
14. `store_config` (`bool`): Whether to store configuration, fidelities, and seed for each evaluation. It consumes much more storage when you use it for large-scale experiments.

## Simulation Using Only the Main Process
This is the description for `ask_and_tell=True`.
This class wraps not only a function but also an optimizer so that we can control the right timing of the addition of data to the optimizer and of job allocation.

While `ask_and_tell=False` requires users to wrap objective function and users simply need to pass the wrapped function to the optimizer prepared by users, `ask_and_tell=True` runs the simulation on the application side.
Unlike the other worker wrappers, each objective function will not run in parallel.
Instead, we internally simulate the cumulative runtime for each worker.
For this sake, the provided optimizer must take the so-called **[ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)** interface.
As long as the optimizer takes this interface, arbitrary optimizers can be used for this class.
Please check [examples](examples/ask_and_tell/) to know how to encapsulate an optimizer with an incompatible interface.
The benefits of this option are:
1. Very stable because all function calls are performed sequentially and unexpected behavior due to parallel computation will not happen while preserving the same results from the parallel version,
2. No slow down even when using a large `n_workers`, and
3. No huge memory consumption caused by multiple in-memory tabular or surrogate datasets.

You can run the example via:
```shell
$ python -m examples.ask_and_tell.random --bench_name branin --seed 0 --n_workers 4
```

## Citation

Please use the following format for the citation of this repository:

```
@article{watanabe2023mfo-simulator,
  title   = {{P}ython Wrapper for Simulating Multi-Fidelity Optimization on {HPO} Benchmarks without Any Wait},
  author  = {S. Watanabe},
  journal = {arXiv:2305.17595},
  year    = {2023},
}
```
