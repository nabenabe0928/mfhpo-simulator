# A Simulator for Multi-Fidelity or Parallel Optimization Using Tabular or Surrogate Benchmarks

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

## Motivation

When we run parallel optimization experiments using tabular or surrogate benchmarks, each evaluation must be ordered based on the runtime that each configuration, in reality, takes.
However, since the evaluation of tabular or surrogate benchmarks, by design, do not take long.
For this reason, the timing each configuration is taken into account must be ordered as if we evaluated each configuration.

In this package, we automatically sort out this problem by pending to pass the hyperparameter configurations to be evaluated internally and in turn, we obtain the right order of each hyperparameter configurations to evaluated.

## Setup

The installation is easily done by `pip-install`:

```shell
$ pip install mfhpo-simulator
```

## Test

If you would like to know the examples of the usage, please clone the whole repository:

```shell
$ git clone mfhpo-simulator
$ pip install -r requirements-for-developer.txt
```

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
1. `--seed` (`int`): The random seeds to be used.
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
Here, we describe the arguments of `CentralWorkerManager`:
1. `subdir_name` (`str`): The directory to store the information.
2. `n_workers` (`int`): The number of parallel workers.
3. `obj_func` (`ObjectiveFuncType`): The objective function to be wrapped. See [`ObjectiveFuncType`](https://github.com/nabenabe0928/mfhpo-simulator/blob/main/benchmark_simulator/_constants.py#L10-L43) for more details.
4. `n_actual_evals_in_opt` (`int`): The number of evaluations inside the optimiziers (this argument will be used only for raising an error).
5. `n_evals` (`int`): The number of evaluations to be stored in the information.
6. `continual_max_fidel` (`Optional[int]`): The maximum fidelity value used for the continual evaluation (it is valid only if we have a single fidelity). If `None`, we just do a normal asynchronous or multi-fidelity optimization. Note that continual evaluation is to train each hyperparameter configuration from scratch or from intermediate results. For example, when we have a train result of a neural network with a hyperparameter configuration `A` for 10 epochs, we train a neural network with `A` for 30 epochs from 10 epochs rather than from scratch,
7. `obj_keys` (`List[str]`): The list of objective names in the output from `obj_func`.
8. `runtime_key` (`str`): The key is for runtime. The output of objective function must include runtime.
9. `obj_keys` (`List[str]`): The list of fidelity names that will be feeded to the objective function. and
10. `seed` (`Optional[List[int]]`): The list of seeds used in each worker.

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
