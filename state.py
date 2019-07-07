from collections import Counter

import numpy as np


class StateBase:
    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)})'


class AtomType(StateBase):
    def __init__(self, symbol: str, atomic_number: int, atomic_weight: float):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight

    def __str__(self):
        return self.symbol


ATOM_TYPES = [
    AtomType('H', 1, 1.007),
    AtomType('C', 6, 12.0096),
    AtomType('N', 7, 14.006),
    AtomType('O', 8, 15.999),
    AtomType('F', 9, 18.998)
]

# adapted From http://hydra.vcp.monash.edu.au/modules/mod2/bondlen.html
BOND_LENGTH_THRESHOLDS = {
    'HH': 1.1,
    'CH': 1.2,
    'HN': 1.2,
    'CC': 1.7,
    'CO': 1.6,
    'HO': 1.3,
    'CN': 1.6,
    'NO': 1.6,
    'NN': 1.6,
    'OO': 1.6,
    'CF': 1.5,
    'FF': None,
    'FH': None,
    'FO': None,
    'FN': None
}

ALLOWED_BONDS_COUNT = {
    'H': (1, 1),
    'C': (2, 4),
    'N': (1, 4),  # it seems nitrogen can have a formal charge in some of these examples
    'O': (1, 2),
    'F': (1, 1)
}


def length(v: np.ndarray) -> float:
    return np.sum(v ** 2) ** 0.5


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return length(a - b)


class Atom(StateBase):
    __slots__ = 'atom_type', 'position', 'bonds'

    def __init__(self, atom_type: AtomType, position: np.ndarray):
        self.atom_type = atom_type
        self.position = position
        self.bonds = []

    def __str__(self):
        return self.hybrdized_symbol

    @property
    def n_bonds(self):
        return len(self.bonds)

    @property
    def hybrdized_symbol(self):
        return self.atom_type.symbol + str(self.n_bonds)

    @property
    def bonded_neighbors(self):
        for bond in self.bonds:
            yield bond.other(self)

    @property
    def bonded_neighbors_count(self):
        return Counter(a.hybrdized_symbol for a in self.bonded_neighbors)

    @property
    def secondary_neighbors(self):
        for neighbor1 in self.bonded_neighbors:
            for neighbor2 in neighbor1.bonded_neighbors:
                if neighbor2 is not self:
                    yield neighbor1, neighbor2

    @property
    def secondary_bonded_neighbors_count(self):
        return Counter((a1.hybrdized_symbol, a2.hybrdized_symbol)
                       for a1, a2 in self.secondary_neighbors)

    @property
    def cycles(self, max_depth=8):
        def rec(path):
            for next in path[-1].bonded_neighbors:
                if next is self:
                    yield path
                if next not in path and len(path) < max_depth:
                    yield from rec(path + (next,))

        yield from rec((self,))


class Bond(StateBase):
    __slots__ = 'a', 'b'

    def __init__(self, a: Atom, b: Atom):
        self.a = a
        self.b = b

    def other(self, atom: Atom) -> Atom:
        if self.a is atom:
            return self.b
        elif self.b is atom:
            return self.a
        else:
            raise ValueError("atom not in bond")

    @property
    def length(self) -> float:
        return distance(self.a.position, self.b.position)

    def __str__(self):
        return f'{self.a}|{self.b}'


class Molecule(StateBase):
    __slots__ = 'name', 'atoms', '_molecular_weight'

    def __init__(self, name: str, atoms: list):
        self.name = name
        self.atoms = atoms
        self._molecular_weight = None

    def __str__(self):
        return self.hybridized_signature

    @property
    def molecular_weight(self):
        if self._molecular_weight is None:
            self._molecular_weight = sum(atom.atom_type.atomic_weight for atom in self.atoms)
        return self._molecular_weight

    @property
    def signature(self):
        return ','.join(f'{s}_{n}' for s, n in
                        sorted(Counter(a.atom_type.symbol for a in self.atoms)
                               .items()))

    @property
    def hybridized_signature(self):
        return ','.join(f'{s}_{n}' for s, n in
                        sorted(Counter(a.hybrdized_symbol for a in self.atoms)
                               .items()))
