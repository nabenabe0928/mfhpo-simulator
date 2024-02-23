# A Simulator for Multi-Fidelity or Parallel Optimization Using Tabular or Surrogate Benchmarks

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

## Motivation

When we run parallel optimization experiments using tabular or surrogate benchmarks, each evaluation must be ordered based on the runtime that each configuration, in reality, takes.
However, the evaluation of tabular or surrogate benchmarks, by design, does not take long.
For this reason, the timing of each configuration taken into account must be ordered as if we evaluated each configuration.

<table>
    <tr>
        <td><img src="figs/compress-conceptual.png" alt=""></td>
    </tr>
</table>

In this package, we automatically sort out this problem by pending to pass the hyperparameter configurations to be evaluated internally, and in turn, we obtain the right order of each hyperparameter configuration to be evaluated.
If the optimizer interface is the [ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) interface, users can pass the optimizer to a simulator directly and the simulator automatically performs the optimization loop as if function calls are run in parallel.

<table>
    <tr>
        <td><img src="figs/api-conceptual.png" alt=""></td>
    </tr>
</table>

> [!TIP]
> For more details, please check out the following documents as well!
> - [Usage and the Difference between with or without Our Wrapper](docs/usage.md)
> - [Tests Using Various Open Source Optimizers](docs/examples.md)
> - [Arguments Used for Our Wrapper/Attributes Provided for Users](docs/wrapper.md)
> - [Simulation Using Only the Main Process (Ask-and-Tell)](docs/ask-and-tell.md)

| Arguments | What Wrapper | Function Call | Requirements | Benefits | Downsides |
|--|:--:|:--:|--|--|--|
| Default | Function  | Parallel | Optimizer spawns child threads or processes (e.g. [DEHB](examples/dehb.py) and [SMAC3](examples/smac.py)) | No need to change the optimizer interface and reproduce exactly how optimizers run | Could be very slow, unstable, and memory-intensive with a large `n_workers` |
|`launch_multiple_wrappers_from_user_side=True`  | Function  | Parallel | Optimizer uses Single-Program Multiple-Data (SPMD) such as MPI, or file-based or server-based synchronization (e.g. [NePS](examples/neps.py) and [BOHB](examples/bohb.py)) | Same above | Same above |
|`ask_and_tell=True` | Function and Optimizer | *Sequential | Optimizer must take the [ask-and-tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) interface (see [example](examples/ask_and_tell/)) | Fast, stable, and memory-efficient even with a large `n_workers` | Force the ask-and-tell interface, may unexpectedly ignore the memory bottleneck that could be caused by parallel runs, and not reproduce the random seed effect |

\* It runs function call sequentially, but function calls are internally processed as if they are run in parallel.

While users do not have to change the interface of optimizers for `ask_and_tell=False` and only need to change the interface of objective function, users may need to change the interface of both for `ask_and_tell=True`.
In principle, `ask_and_tell=True` requires optimizers to be the ask-and-tell interface.
In exchange for the strict constraint, it stabilizes the simulation.

> [!NOTE]
> Our wrapper assumes that none of the workers will not die and any additional workers will not be added after the initialization.
> Therefore, if any workers die, our current wrapper hangs and keeps warning except we provide `max_waiting_time` for the instantiation.
> Furthermore, our package **cannot be run on Windows OS** because the Python module `fcntl` is not supported on Windows OS.
> Although our package supports MacOS, it is advisable to use Linux system.

## Setup

The installation is easily done by `pip-install`:

```shell
$ pip install mfhpo-simulator
```

The requirements are:
- Unix system
- Python 3.8 or later

The dependencies of this package are only **numpy** and **ujson**.

Basic usage is available at [Usage and the Difference between with or without Our Wrapper](docs/usage.md) and [Tests Using Various Open Source Optimizers](docs/examples.md)

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
