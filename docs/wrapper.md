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
11. `fidel_keys` (`list[str]`): The list of fidelity names that will be fed to the objective function,
12. `seed` (`int | None`): The random seed to be used in each worker,
13. `max_waiting_time` (`float`): The maximum waiting time for each worker. If workers wait for the provided amount of time, the wrapper will return only `INF`,
14. `store_config` (`bool`): Whether to store configuration, fidelities, and seed for each evaluation. It consumes much more storage when you use it for large-scale experiments, and
15. `check_interval_time` (`float`): How often each worker should check whether a new job can be assigned to it. For example, if `1e-2` is specified, each worker check whether they can get a new job every `1e-2` seconds. If there are many workers, too small `check_interval_time` may cause a big bottleneck. On the other hand, a big `check_interval_time` spends more time for waiting. By default, `check_interval_time` is set to a relatively small number, so users might rather want to increase the number to avoid the bottleneck for many workers.

## Attributes Provided for Users

### Instance Variables

1. `dir_name` (`str`): The relative path where results will be stored and it returns `./mfhpo-simulator/<save_dir_name>`,
2. `obj_keys` (`list[str]`): The objective names that will be collected in results and the returned dict from users' objective functions must contain these keys. If you want to include the runtime in the results, you also need to include the runtime_key in obj_keys,
3. `runtime_key` (`str`): The runtime name that will be used to define the runtime which the user objective function really took. The returned dict from users' objective functions must contain this key,
4. `fidel_keys` (`list[str]`): The fidelity names that will be used in users' objective functions. `fidels` passed to the objective functions must contain these keys. When `continual_max_fidel=True`, fidel_keys can contain only one key and this fidelity will be used for the definition of continual learning,
5. `n_actual_evals_in_opt` (`int`): The number of configuration evaluations during the actual optimization. Note that even if some configurations are continuations from an existing config with lower fidelity, they are counted as separated config evaluations, and
6. `n_workers` (`int`): The number of workers used in the user-defined optimizer.

`obj_keys`, `runtime_key`, and `fidel_keys` are necessary to match the signature of user-defined objective functions with our API.


### Methods

1. `__call__(self, eval_config: dict[str, Any], *, fidels: dict[str, int | float] | None = None, **data_to_scatter: Any) -> dict[str, float]`

The wrapped objective function used in the user-defined optimizer and valid only if `ask_and_tell=False`.

`eval_config` is the configuration that will be passed to the objective function via our wrapper, but the wrapper does not use this information and it will be used only if `store_config=True` for the storage purpose.

`fidels` is the fidelity parameters that will be passed to the objective function via our wrapper.
If `continual_max_fidel=True`, `fidels` must contain only one name and our wrapper uses the value in `fidels` for continual learning.
Otherwise, our API will not use the information in `fidels` except for the storage purpose for `store_config=True`.

`data_to_scatter` is any information that will be passed to the objective function.
It is especially important when users would like to scatter in-memory large-size data using such as `dask.scatter` because parallel processing in optimizers usually requires serialization and deserialization of the objective function.
We can simply avoid this issue by making the data size of the objective function as small as possible and passing the (in-memory) data to the objective function when it is called.

The return value of the method is `dict[str, float]` where the keys are the union of `[runtime_key]` and `obj_keys` and the values are their corresponding values.

2. `simulate(self, opt: AbstractAskTellOptimizer) -> None`

The optimization loop for the wrapped objective function and the user-defined optimizer and valid only if `ask_and_tell=True`.

Users can simply start the simulation of an optimization using `opt` and `obj_func` via `simulate`.