import os
import os.path

import pandas as pd


N_PARTITIONS = 100


def partition(name):
    print('partitioning', name)
    data = pd.read_csv(f'data/{name}.csv')
    partition_indices = data['molecule_name'].map(hash) % N_PARTITIONS
    for index, df in data.groupby(partition_indices):
        print('writing', name, index)
        path = f'data/partitions/{name}/{index}.p'
        dr = os.path.dirname(path)
        os.makedirs(dr, exist_ok=True)
        df.to_pickle(path)


def main():
    partition('train')
    partition('test')
    partition('structures')

__name__ == '__main__' and main()
