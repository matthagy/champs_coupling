import multiprocessing as mp
from multiprocessing.pool import ApplyResult
from time import sleep

from tqdm import tqdm


def mp_map(func, args):
    with mp.Pool() as pool:
        futures = [pool.apply_async(func, args_i) for args_i in args]
        with tqdm(total=len(futures)) as t:
            while futures:
                new_futures = []
                for future in futures:
                    future: ApplyResult = future
                    if future.ready():
                        future.get()
                        t.update()
                    else:
                        new_futures.append(future)
                futures = new_futures
                sleep(0.25)


def mp_map_parititons(func):
    from partition_by_molecule import N_PARTITIONS
    mp_map(func, ((partition_index,) for partition_index in range(N_PARTITIONS)))
