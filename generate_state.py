import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from util import mp_map
from state import ATOM_TYPES, BOND_LENGTH_THRESHOLDS, ALLOWED_BONDS_COUNT, Atom, Bond, Molecule
from partition_by_molecule import N_PARTITIONS

atom_type_by_symbol = {atom_type.symbol: atom_type for atom_type in ATOM_TYPES}


def compute_bonds(molecule):
    atoms = molecule.atoms
    for i, ai in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            aj = atoms[j]
            d = ((ai.position - aj.position) ** 2).sum() ** 0.5
            tp = ''.join(sorted([ai.atom_type.symbol, aj.atom_type.symbol]))
            t = BOND_LENGTH_THRESHOLDS[tp]
            if t is not None and d < t:
                bond = Bond(ai, aj)
                ai.bonds.append(bond)
                aj.bonds.append(bond)

    for i, ai in enumerate(atoms):
        mn, mx = ALLOWED_BONDS_COUNT[ai.atom_type.symbol]
        if not (mn <= ai.n_bonds <= mx):
            print(f'bad number of bonds {ai.n_bonds} for {i} in {molecule.name} {ai.atom_type.symbol}')


def main():
    structures = pd.read_csv('data/structures.csv')
    molecules = []
    for name, molecule_df in tqdm(structures.groupby('molecule_name')):
        molecule_df = molecule_df.set_index('atom_index').sort_index()
        atoms = []
        for atom_index, atom_row in molecule_df.iterrows():
            atom_type = atom_type_by_symbol[atom_row['atom']]
            position = np.array([atom_row['x'], atom_row['y'], atom_row['z']])
            atom = Atom(atom_type, position)
            atoms.append(atom)
        molecule = Molecule(name, atoms)
        compute_bonds(molecule)
        molecules.append(molecule)

    with open('data/molecules.p', 'wb') as fp:
        pickle.dump(molecules, fp)


__name__ == '__main__' and main()
