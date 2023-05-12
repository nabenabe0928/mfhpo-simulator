# A Simulator for Multi-Fidelity or Parallel Optimization Using Tabular or Surrogate Benchmarks

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

## Motivation

When we run parallel optimization experiments using tabular or surrogate benchmarks, each evaluation must be ordered based on the runtime that each configuration, in reality, takes.
However, since the evaluation of tabular or surrogate benchmarks, by design, do not take long.
For this reason, the timing each configuration is taken into account must be ordered as if we evaluated each configuration.

In many papers, 

## Setup & test

1. Install the package

```shell
$ pip install mfhpo-simulator
```

2. Save the following file (`run_test.py`)

```python
from benchmark_simulator import ...
```

3. Run the Python file

```shell
$ python run_test.py
```

## Usage
