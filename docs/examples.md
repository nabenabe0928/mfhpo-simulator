## Tests Using Various Open Source Optimizers

The very minimal examples are provided in [examples/minimal/](../examples/minimal/).
After `pip install -r requirements-for-developer.txt`, you can run these examples by:

```shell
# DEHB
$ python -m examples.minimal.dehb

# BOHB
$ python -m examples.minimal.bohb

# SMAC (It conflicts with the others, so you need to do `pip install -r requirements-for-smac.txt`)
$ python -m examples.minimal.smac

# NePS
$ ./examples/minimal/neps.sh --n_workers 4
```

Note that the environment for SMAC3 is separately defined due to some dependency issues:

```shell
$ pip install -r requirements-for-smac.txt
```

Additionally, we need to increase the `retries` in the following files:
- intensifier/intensifier.py
- facade/abstract_facade.py
- main/config_selector.py
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
