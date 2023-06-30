from contextlib import contextmanager
import multiprocessing


@contextmanager
def get_pool(n_workers: int) -> multiprocessing.Pool:
    pool = multiprocessing.Pool(processes=n_workers)
    yield pool
    pool.close()
    pool.join()
