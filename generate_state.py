import os
import os.path
import pickle

import numpy as np
import pandas as pd
import openbabel


from state import ATOM_TYPES, BOND_LENGTH_THRESHOLDS, ALLOWED_BONDS_COUNT, Atom, Bond, Molecule
from util import mp_map_parititons

atom_type_by_symbol = {atom_type.symbol: atom_type for atom_type in ATOM_TYPES}


def compute_bonds(molecule: Molecule):
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


def compute_partial_charges(molecule: Molecule):
    ob_mol = openbabel.OBMol()
    ob_atoms = []
    for atom in molecule.atoms:
        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(atom.atom_type.atomic_number)
        ob_atoms.append(ob_atom)

    bonds = {b for a in molecule.atoms for b in a.bonds}
    for bond in bonds:
        ob_a1 = ob_atoms[molecule.atoms.index(bond.a)]
        ob_a2 = ob_atoms[molecule.atoms.index(bond.b)]
        ob_bond = ob_mol.NewBond()
        ob_bond.SetBegin(ob_a1)
        ob_bond.SetEnd(ob_a2)

    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    charge_model.ComputeCharges(ob_mol)

    for atom, ob_atom in zip(molecule.atoms, ob_atoms):
        atom.partial_charge = ob_atom.GetPartialCharge()


def compute_molecule(name, molecule_df):
    molecule_df = molecule_df.set_index('atom_index').sort_index()
    atoms = []
    for atom_index, atom_row in molecule_df.iterrows():
        atom_type = atom_type_by_symbol[atom_row['atom']]
        position = np.array([atom_row['x'], atom_row['y'], atom_row['z']])
        atom = Atom(atom_type, position)
        atoms.append(atom)
    molecule = Molecule(name, atoms)

    compute_bonds(molecule)
    compute_partial_charges(molecule)

    return molecule


def process_partition(index):
    structures = pd.read_pickle(f'data/partitions/structures/{index}.p')
    molecules = []
    for name, molecule_df in structures.groupby('molecule_name'):
        molecules.append(compute_molecule(name, molecule_df))

    path = f'data/partitions/molecules/{index}.p'
    dr = os.path.dirname(path)
    if not os.path.exists(dr):
        os.makedirs(dr, exist_ok=True)

    with open(path, 'wb') as fp:
        pickle.dump(molecules, fp)


def main():
    mp_map_parititons(process_partition)


__name__ == '__main__' and main()
