import multiprocessing
import os
import shutil
import sys
from contextlib import contextmanager

from benchmark_simulator._constants import DIR_NAME


SUBDIR_NAME = "dummy"
IS_LOCAL = eval(os.environ.get("MFHPO_SIMULATOR_TEST", "False"))
ON_UBUNTU = sys.platform == "linux"
DIR_PATH = os.path.join(DIR_NAME, SUBDIR_NAME)


def remove_tree():
    try:
        shutil.rmtree(DIR_PATH)
    except FileNotFoundError:
        pass


def get_n_workers():
    n_workers = 4 if IS_LOCAL else 2  # github actions has only 2 cores
    return n_workers


@contextmanager
def get_pool(n_workers: int) -> multiprocessing.Pool:
    pool = multiprocessing.Pool(processes=n_workers)
    yield pool
    pool.close()
    pool.join()
