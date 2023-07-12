import itertools
import json
import subprocess
import time

import numpy as np


def get_command(opt: str, seed: int, mode: str, deterministic: bool) -> str:
    cmd = [
        f"python -m demo.validation_with_{opt} --raw True",
        f"--seed {seed}",
        f"--mode {mode}",
        f"--deterministic {deterministic}",
    ]
    return " ".join(cmd)


def get_file_name(opt: str, seed: int, mode: str, deterministic: bool) -> str:
    return f"demo/{opt}_{mode}_noise={not deterministic}/{seed:0>3}.json"


for opt, deterministic, seed, mode in itertools.product(
    *(
        ["random", "optuna"],
        [True, False],
        range(10),
        ["single", "multi", "no"],
    )
):
    cmd = get_command(opt, seed, mode, deterministic)
    start = time.time_ns()
    subprocess.call(cmd, shell=True)
    end = time.time_ns()
    elapsed_time = (end - start) / 10**9
    print(f"Execution time: {elapsed_time:.4f} seconds / {cmd}")

    file_name = get_file_name(opt, seed, mode, deterministic)
    with open(file_name, mode="r") as f:
        data = json.load(f)

    actual_cumtime = data["actual_cumtime"]
    misc_time = elapsed_time - actual_cumtime[-1]
    data["actual_cumtime"] = (np.array(actual_cumtime) + misc_time).tolist()
    with open(file_name, mode="w") as f:
        json.dump(data, f, indent=4)
