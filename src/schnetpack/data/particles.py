import numbers
import numpy as np
from ase import Atoms
from schnetpack.data.particle import Particle

"""Definition of the Particle and Particles classes, which extend the 
ASE Atom and Atoms with typing, molecule membership, and connectivity, 
to generealize the description of particle-based bulk systems made of 
Coarse Grained moieties.
"""

class Particles(Atoms):
    """Particles object.

    The Particles object extends the ASE Atoms one, and, analogously, it
    can represent an isolated molecule, or a periodically repeated 
    structure. All the Atoms attributes may be defined for Particles
    as well. Information is stored in ndarrays.
    
    Particles has a unit cell and there may be periodic boundary conditions
    along any of the three unit cell axes.

    In order to calculate energies, forces and stresses, a Calculator
    object has to attached to the Particles object.

    Attributes added to Particles:

    types: list of int
        List of particle types, of length = total number of particles 
    molecule_ids: list of int
        List of molecule membership isd, of length = total number of particles 
    bonds_list: list of list of int
        List of length = number of bonds and each element is a 2 element 
        list containing the id of the bonded particles. Each pair should
        should be given only once. 
    angles_list: list of list of int
        List of length = number of angles and each element is a 3 element 
        list containing the id of the particles defining the angle. Each 
        triplet should be given only once. 
    bond_types: list of int
        List of bond types correspnding to the elements in bonds_list,
        of length = total number of particles 
    angle_types: list of int
        List of angle types correspnding to the elements in bonds_list,
        of length = total number of particles 

    The methods referencing Atoms have been overridden to reference 
    Particle instead.
    """
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None,
                 
                 particle_types=None,
                 molecule_ids=None,
                 bonds_list=None,
                 angles_list=None,
                 bond_types=None,
                 angle_types=None, 
                 
                 n_bonds = [0],
                 n_angles = [0],):

        super().__init__(symbols,
                 positions, numbers, tags, momenta, 
                 masses, magmoms, charges,                 
                 scaled_positions, cell, pbc, celldisp,                 
                 constraint, calculator, info, velocities)

        self.set_molecule_ids(molecule_ids)
        self.set_particle_types(particle_types)

        self.set_bonds_list(bonds_list)
        self.set_angles_list(angles_list)
        self.set_bond_types(bond_types)
        self.set_angle_types(angle_types)

        self.set_n_bonds(n_bonds)
        self.set_n_angles(n_angles)

    def set_n_bonds(self, n_bonds = [0]):
        """Set n_bonds."""
        if len(self.arrays["bonds_list"])>0:
            n_bonds = [len(self.arrays["bonds_list"])//2]
        self.set_array('n_bonds', n_bonds, int)

    def get_n_bonds(self):
        """Get n_bonds."""
        if 'n_bonds' in self.arrays:
            return self.arrays['n_bonds'].copy()
        else:
            raise RuntimeError("Trying to get n_bonds, but it was not defined")
        
    def set_n_angles(self, n_angles = [0]):
        """Set n_angles."""
        if len(self.arrays["angles_list"])>0:
            n_angles = [len(self.arrays["angles_list"])//3]
        self.set_array('n_angles', n_angles, int)

    def get_n_angles(self):
        """Get n_angles."""
        if 'n_angles' in self.arrays:
            return self.arrays['n_angles'].copy()
        else:
            raise RuntimeError("Trying to get n_angles, but it was not defined")

    def set_angle_types(self, angle_types=None):
        """Set the angle_types."""

        if angle_types is None:
            self.set_array('angle_types', None)
        elif len(angle_types) == 0:
            angle_types = [ [] for _ in range(len(self.arrays['numbers']))]
            self.set_array('angle_types', angle_types, int)
        else:
            angle_types = np.asarray(angle_types)
            self.set_array('angle_types', angle_types, int)


    def get_angle_types(self):
        """Get the angle_types."""
        if 'angle_types' in self.arrays:
            return self.arrays['angle_types'].copy()
        else:
            raise RuntimeError("Trying to get angle_types, but they were not defined")


    def set_bond_types(self, bond_types=None):
        """Set the bond_types."""

        if bond_types is None:
            self.set_array('bond_types', None)
        elif len(bond_types) == 0:
            bond_types = [ [] for _ in range(len(self.arrays['numbers']))]
            self.set_array('bond_types', bond_types, int)
        else:
            bond_types = np.asarray(bond_types)
            self.set_array('bond_types', bond_types, int)


    def get_bond_types(self):
        """Get the bond_types."""
        if 'bond_types' in self.arrays:
            return self.arrays['bond_types'].copy()
        else:
            raise RuntimeError("Trying to get bond_types, but they were not defined")


    def set_angles_list(self, angles_list=None):
        """Set the angles_list."""

        if angles_list is None:
            self.set_array('angles_list', None)
        elif len(angles_list) == 0:
            angles_list = [ [] for _ in range(len(self.arrays['numbers']))]
            self.set_array('angles_list', angles_list, int)
        else:
            angles_list = np.asarray(angles_list)
            self.set_array('angles_list', angles_list, int)


    def get_angles_list(self):
        """Get the angles_list."""
        if 'angles_list' in self.arrays:
            return self.arrays['angles_list'].copy()
        else:
            raise RuntimeError("Trying to get angles_list, but they were not defined")


    def set_bonds_list(self, bonds_list=None):
        """Set the bonds_list."""

        if bonds_list is None:
            self.set_array('bonds_list', None)
        elif len(bonds_list) == 0:
            bonds_list = [ [] for _ in range(len(self.arrays['numbers']))]
            self.set_array('bonds_list', bonds_list, int)
        else:
            bonds_list = np.asarray(bonds_list)
            self.set_array('bonds_list', bonds_list, int)


    def get_bonds_list(self):
        """Get the bonds_list."""
        if 'bonds_list' in self.arrays:
            return self.arrays['bonds_list'].copy()
        else:
            raise RuntimeError("Trying to get bonds_list, but they were not defined")


    def set_molecule_ids(self, molecule_ids=None):
        """Set the molecule_ids."""

        if molecule_ids is None:# or len(molecule_ids) == 0:
            self.set_array('molecule_ids', None)
        else:
            molecule_ids = np.asarray(molecule_ids)
            self.set_array('molecule_ids', molecule_ids, int)


    def get_molecule_ids(self):
        """Get the molecule_ids."""
        if 'molecule_ids' in self.arrays:
            return self.arrays['molecule_ids'].copy()
        else:
            raise RuntimeError("Trying to get molecule_ids, but they were not defined")


    def set_particle_types(self, particle_types=None):
        """Set the molecule_ids."""

        if particle_types is None:# or len(particle_types) == 0: 
            self.set_array('particle_types', None)
        else:
            particle_types = np.asarray(particle_types)
            self.set_array('particle_types', particle_types, int)


    def get_particle_types(self):
        """Get the molecule_ids."""
        if 'particle_types' in self.arrays:
            return self.arrays['particle_types'].copy()
        else:
            raise RuntimeError("Trying to get particle_types, but they were not defined")

    def extend(self, other):
        """Extend atoms object by appending atoms from *other*."""
        if isinstance(other, Particle):
            other = self.__class__([other])

        n1 = len(self)
        n2 = len(other)

        for name, a1 in self.arrays.items():
            a = np.zeros((n1 + n2,) + a1.shape[1:], a1.dtype)
            a[:n1] = a1
            if name == 'masses':
                a2 = other.get_masses()
            else:
                a2 = other.arrays.get(name)
            if a2 is not None:
                a[n1:] = a2
            self.arrays[name] = a

        for name, a2 in other.arrays.items():
            if name in self.arrays:
                continue
            a = np.empty((n1 + n2,) + a2.shape[1:], a2.dtype)
            a[n1:] = a2
            if name == 'masses':
                a[:n1] = self.get_masses()[:n1]
            else:
                a[:n1] = 0

            self.set_array(name, a)     


    def __getitem__(self, i):
        """Return a subset of the atoms.

        i -- scalar integer, list of integers, or slice object
        describing which atoms to return.

        If i is a scalar, return an Atom object. If i is a list or a
        slice, return an Atoms object with the same cell, pbc, and
        other associated info as the original Atoms object. The
        indices of the constraints will be shuffled so that they match
        the indexing in the subset returned.

        """

        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return Particle(atoms=self, index=i)
        elif not isinstance(i, slice):
            i = np.array(i)
            # if i is a mask
            if i.dtype == bool:
                if len(i) != len(self):
                    raise IndexError('Length of mask {} must equal '
                                     'number of atoms {}'
                                     .format(len(i), len(self)))
                i = np.arange(len(self))[i]

        import copy

        conadd = []
        # Constraints need to be deepcopied, but only the relevant ones.
        for con in copy.deepcopy(self.constraints):
            try:
                con.index_shuffle(self, i)
            except (IndexError, NotImplementedError):
                pass
            else:
                conadd.append(con)

        atoms = self.__class__(cell=self.cell, pbc=self.pbc, info=self.info,
                               # should be communicated to the slice as well
                               celldisp=self._celldisp)
        # TODO: Do we need to shuffle indices in adsorbate_info too?

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a[i].copy()

        atoms.constraints = conadd
        return atoms


    def __eq__(self, other):
        """Check for identity of two Particle objects.

        Identity means: same positions, atomic numbers, unit cell and
        periodic boundary conditions."""
        if not isinstance(other, Particle):
            return False
        a = self.arrays
        b = other.arrays
        return (len(self) == len(other) and
                (a['positions'] == b['positions']).all() and
                (a['numbers'] == b['numbers']).all() and
                (self.cell == other.cell).all() and
                (self.pbc == other.pbc).all() and
                (a['molecule_ids'] == b['molecule_ids']).all() and
                (a['particle_types'] == a['molecule_ids']).all())


    def new_array(self, name, a, dtype=None, shape=None):
        """Add new array.

        Overridden because the parent method operates under the assumption that 
        every new array should have an entry per atom, which is not the case
        for the connectivity properties."""

        if dtype is not None:
            a = np.array(a, dtype, order='C')
            if len(a) == 0 and shape is not None:
                a.shape = (-1,) + shape
        else:
            if not a.flags['C_CONTIGUOUS']:
                a = np.ascontiguousarray(a)
            else:
                a = a.copy()

        if name in self.arrays:
            raise RuntimeError('Array {} already present'.format(name))

        # for b in self.arrays.values():
        #    if len(a) != len(b):
        #        raise ValueError('Array "%s" has wrong length: %d != %d.' %
        #                         (name, len(a), len(b)))
        #    break

        if shape is not None and a.shape[1:] != shape:
            raise ValueError('Array "%s" has wrong shape %s != %s.' %
                             (name, a.shape, (a.shape[0:1] + shape)))

        self.arrays[name] = a