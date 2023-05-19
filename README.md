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
$ python -m examples.smac --seed 0 --dim 0 --bench_name hartmann --n_workers 4
```

Each argument is defined as follows:
1. --seed (int): The random seeds to be used.
2. --bench_name (Literal["hpolib", "jahs", "lc", "hartmann", "branin"]): The benchmark to be used.
3. --n_workers (int): The number of parallel workers to be used. Too high numbers may crash your system because the specified benchmark dataset must stay on the memory for each process/thread.
4. --dataset_id (Optional[int]): The dataset ID to be used in the specified dataset (0 to 3 for HPOlib, 0 to 33 for LCBench, and 0 t0 2 for JAHS-Bench-201). The default value is 0.
5. --dim (Optional[int]): The dimensionality of the Hartmann function and it is used only Hartmann function. The default value is 3.

Note that `--seed` does not guarantee the reproducitility because of the parallel processing nature.

## Citation

Please use the following format for the citation of this repository:

```
@inproceedings{watanabe2023pareto,
  title   = {{P}ython Wrapper for Simulating Asynchronous Optimization on Cheap Benchmarks without Any Wait},
  author  = {S. Watanabe},
  year    = {2023},
  journal = {arXiv:XXXX.XXXXX}
}
```
