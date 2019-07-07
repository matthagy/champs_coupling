import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from state import Molecule


def compute_neighbors_for_molecule(m: Molecule):
    n = len(m.atoms)

    atom_distances = np.zeros((n, n), dtype=float)
    for i, ai in enumerate(m.atoms):
        for j in range(i + 1, n):
            aj = m.atoms[j]
            d = np.sum((ai.position - aj.position) ** 2) ** 0.5
            atom_distances[i, j] = d
            atom_distances[j, i] = d

    atom_types = [a.hybridized_symbol for a in m.atoms]
    di, dj = np.where((atom_distances > 0) & (atom_distances < 1.5))
    neighbor_counts = defaultdict(Counter)
    for i, j in zip(di, dj):
        neighbor_counts[i][atom_types[j]] += 1
    return neighbor_counts


print('loading molecules')
with open('data/molecules.p', 'rb') as fp:
    molecules = pickle.load(fp)

#molecules = molecules[:100]

mol_neighbor_counts = {}
with tqdm(molecules) as t:
    for m in t:
        mol_neighbor_counts[m.name] = compute_neighbors_for_molecule(m)

with open('data/molecule_neighbor_counts.p', 'wb') as fp:
    pickle.dump(mol_neighbor_counts, fp, pickle.HIGHEST_PROTOCOL)


