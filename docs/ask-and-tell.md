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
# Random search
$ python -m examples.ask_and_tell.random --bench_name branin --seed 0 --n_workers 4

# Optuna
$ python -m examples.ask_and_tell.optuna --bench_name branin --seed 0 --n_workers 4
```
