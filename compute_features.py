import pickle
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

from state import Molecule, Atom, Bond, length, distance


def main():
    print('loading molecules')
    with open('data/molecules.p', 'rb') as fp:
        molecules = pickle.load(fp)
    molecules_by_name = {m.name: m for m in molecules}

    compute_features('train', molecules_by_name)
    compute_features('test', molecules_by_name)


def compute_features(name, molecules_by_name):
    print('computing features for', name)
    data = pd.read_csv(f'data/{name}.csv')

    for coupling_type, type_df in data.groupby('type'):
        print('coupling', coupling_type)

        type_df = type_df.sort_values('id')

        with tqdm(total=len(type_df)) as t:
            with open(f'data/features_{name}_{coupling_type}.p', 'wb') as fp:
                for _, row in type_df.iterrows():
                    t.update()

                    molecule_name = row['molecule_name']
                    molecule: Molecule = molecules_by_name[molecule_name]

                    features = compute_pair_features(row, molecule)
                    features['id'] = row['id']

                    pickle.dump(features, fp, pickle.HIGHEST_PROTOCOL)


def compute_pair_features(row: pd.Series,
                          molecule: Molecule) -> dict:
    a0: Atom = molecule.atoms[row['atom_index_0']]
    a1: Atom = molecule.atoms[row['atom_index_1']]

    features = {'distance': distance(a0, a1),
                'molecular_weight': molecule.molecular_weight,
                'a0_bonds': a0.n_bonds,
                'a1_bonds': a1.n_bonds}

    for prefix, bonds in [('b0', a0.bonded_neighbors_count),
                          ('b1', a1.bonded_neighbors_count)]:
        for bond_type, bond_count in bonds.items():
            features[prefix + '_' + bond_type] = bond_count

    for prefix, bonds in [('bs0', a0.secondary_bonded_neighbors_count),
                          ('bs1', a1.secondary_bonded_neighbors_count)]:
        for (ba1t, ba2t), bond_count in bonds.items():
            features[prefix + '_' + ba1t + '-' + ba2t] = bond_count

    bond_path = find_shortest_path_between_atoms(a0, a1)
    if not bond_path:
        print("couldn't find bond path between atoms")
    else:
        b0: Bond = bond_path[0]
        b1: Bond = bond_path[-1]

        v0 = bond_vector(a0, b0)
        v1 = bond_vector(a1, b1)

        l0 = length(v0)
        l1 = length(v1)

        dot = np.sum(v0 * v1)

        features.update({
            'a0_bond_length': l0,
            'a1_bond_length': l1,
            'bond_vector_dot': dot,
            'bond_vector_dot_norm': dot / (l0 * l1),
            'bond_path_length': sum(b.length for b in bond_path)
        })

    return features


def find_shortest_path_between_atoms(atom_source: Atom,
                                     atom_target: Atom) -> tuple:
    bonds_to_explore = deque((atom_source, (b,))
                             for b in atom_source.bonds)

    while bonds_to_explore:
        prev_atom, prev_bonds = bonds_to_explore.popleft()
        next_atom = prev_bonds[-1].other(prev_atom)
        if next_atom is atom_target:
            return prev_bonds
        else:
            for next_bond in next_atom.bonds:
                if next_bond.other(next_atom) is not prev_atom:
                    bonds_to_explore.append((next_atom,
                                             prev_bonds + (next_bond,)))
    return None


def bond_vector(src: Atom, bond: Bond) -> np.ndarray:
    return src.position - bond.other(src).position


__name__ == '__main__' and main()
