## Usage

We first note that this example is based on [examples/random_with_dummy.py](../examples/random_with_dummy.py).
After `pip install mfhpo-simulator`, you can test it with:

```shell
$ python -m examples.random_with_dummy
```

Suppose a user have the following optimizer and objective function:

```python
from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np


class RandomOptimizer:
    def __init__(self, func: callable, n_workers: int, n_evals: int, value_range: tuple[float, float], seed: int = 42):
        self._func = func
        self._n_workers = n_workers
        self._n_evals = n_evals
        self._rng = np.random.RandomState(seed)
        self._lower, self._upper = value_range

    def optimize(self) -> list[float]:
        pool = multiprocessing.Pool(processes=self._n_workers)

        _results = []
        for i in range(self._n_evals):
            x = self._rng.random() * (self._upper - self._lower) + self._lower
            _results.append(pool.apply_async(self._func, args=[x]))

        pool.close()
        pool.join()
        return [r.get() for r in _results]


def dummy_func(x: float) -> float:
    # NOTE: Let's assume that the actual runtime here is huge, but we can query a pre-recorded result in a fraction
    # e.g. actual_runtime = loss * 1e3
    return x ** 2
```

Then users can run random search as follows:

```python
import time


def dummy_func_wrapper(x: float) -> float:
    loss = dummy_func(x)
    actual_runtime = loss * 1e3
    # NOTE: dummy_func needs to wait internally for the actual runtime to simulate asynchronous optimization
    time.sleep(actual_runtime)
    return loss


results = RandomOptimizer(func=dummy_func_wrapper, n_workers=4, n_evals=100, value_range=(-5.0, 5.0)).optimize()
```

Then our wrapper requires some modifications in signatures of the input of the objective function and the return of our wrapper as follows:

```python
from __future__ import annotations

from typing import Any

from benchmark_simulator import ObjectiveFuncWrapper


def dummy_func_wrapper(eval_config: dict[str, Any], **kwargs) -> dict[str, float]:
    # 0. Adapt the function signature to our wrapper interface
    loss = dummy_func(x=eval_config["x"])
    # For our wrapper, we do not need to wait internally!
    actual_runtime = loss * 1e3
    # Default: obj_keys = ["loss"], runtime_key = "runtime"
    # You can add more keys to obj_keys then our wrapper collects these values as well.
    return dict(loss=loss, runtime=actual_runtime)


class MyObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # 0. Adapt the callable of the objective function to RandomOptimizer interface
    def __call__(self, x: float) -> float:
        results = super().__call__(eval_config={"x": x})
        return results[self.obj_keys[0]]
```

Using our wrapper, the asynchronous optimization could be simply run as follows:

```python
# 1. Define a wrapper instance (Default is n_workers=4, but you can change it from the argument)
wrapper = MyObjectiveFuncWrapper(obj_func=dummy_func_wrapper)

RandomOptimizer(
    # 2. Feed the wrapped objective function to the optimizer directly
    func=wrapper,
    n_workers=wrapper.n_workers,
    n_evals=wrapper.n_actual_evals_in_opt,
    value_range=(-5.0, 5.0),
).optimize()  # 3. just start the optimization!
```

In principle, the only difference is the input of `func` and it was replaced with `wrapper`.
Then our wrapper automatically sorts out the waiting time internally and all the evaluations are available at `mfhpo-simulator/<save_dir_name>/results.json`. 
