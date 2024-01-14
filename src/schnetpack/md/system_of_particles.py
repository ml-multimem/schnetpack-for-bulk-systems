import torch

import numpy as np

from ase import Atoms

from schnetpack.data import Particles
from schnetpack.md import System 
from schnetpack.md.system import SystemWarning
from schnetpack import units as spk_units

from typing import Union, List, OrderedDict

__all__ = ["SystemOfParticles"]

class SystemOfParticles(System):

    def __init__(self):

        super(SystemOfParticles, self).__init__()

        self.register_uninitialized_buffer("particle_types", dtype=torch.long)
        self.register_uninitialized_buffer("molecule_ids", dtype=torch.long)

        self.register_uninitialized_buffer("bonds_list", dtype=torch.long)
        self.register_uninitialized_buffer("angles_list", dtype=torch.long)
        self.register_uninitialized_buffer("bond_types", dtype=torch.long)
        self.register_uninitialized_buffer("angle_types", dtype=torch.long)

        self.register_uninitialized_buffer("n_bonds", dtype=torch.long)
        self.register_uninitialized_buffer("n_angles", dtype=torch.long)

        self.register_uninitialized_buffer("images", dtype=torch.long)


    def load_molecules(
        self,
        molecules: Union[Particles, List[Particles]],
        n_replicas: int = 1,
        position_unit_input: Union[str, float] = "Angstrom",
        mass_unit_input: Union[str, float] = 1.0,
        ):

        if len(molecules) > 1:
            raise RuntimeError("Multiple initial configurations found in the database. Please supply a single initial configuration.")

        # We perform here # 0) of the load_molecules method of the parent class, 
        
        if isinstance(molecules, Particles):
            molecules = [molecules]

        (mol_ids_array, mol_count_array) = np.unique(molecules[0].arrays['molecule_ids'], return_counts=True)
        # Then we call the parent method to populate part of the properties
        super().load_molecules(molecules, n_replicas, position_unit_input, mass_unit_input)

        # Set up unit conversion
        positions2internal = spk_units.unit2internal(position_unit_input)
        mass2internal = spk_units.unit2internal(mass_unit_input)

        # We overwrite with the desired handling of molecules
        # 1) Get number of molecules        
        #self.n_molecules = len(mol_ids_array)
        
        # 2) Construct array with number of atoms in each molecule
        #self.n_atoms = torch.zeros(self.n_molecules, dtype=torch.long)
        #for i in range(self.n_molecules):
        #    self.n_atoms[i] = mol_count_array[i]

        # 3) Construct basic property arrays
        #self.energy = torch.zeros(self.n_replicas, self.n_molecules, 1)

        # Relevant for periodic boundary conditions and simulation cells
        #one_cell = self.cells[0][0]
        #one_pbc = self.pbc[0][0]
        #self.cells = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        #self.pbc = torch.zeros(1, self.n_molecules, 3)

        # 5) Populate arrays according to the data provided in molecules
        #idx_c = 0
        #for i in range(self.n_molecules):
        #    n_atoms = self.n_atoms[i]

            # Aggregation array
        #    self.index_m[idx_c : idx_c + n_atoms] = i

            # Properties for cell simulations
        #    self.cells[:, i, :, :] = one_cell

        #    self.pbc[0, i, :] = one_pbc

        #    idx_c += n_atoms

        # Convert periodic boundary conditions to Boolean tensor
        #self.pbc = self.pbc.bool()

        # We integrate # 5) with the additional properties        
        self.particle_types = torch.from_numpy(molecules[0].get_particle_types()).long()
        self.molecule_ids = torch.from_numpy(molecules[0].get_molecule_ids()).long()
        self.bonds_list = torch.from_numpy(molecules[0].get_bonds_list()).long().reshape((-1, 2))
        self.angles_list = torch.from_numpy(molecules[0].get_angles_list()).long().reshape((-1, 3))
        self.bond_types = torch.from_numpy(molecules[0].get_bond_types()).long()
        self.angle_types = torch.from_numpy(molecules[0].get_angle_types()).long()
        self.n_bonds = torch.from_numpy(molecules[0].get_n_bonds()).long()
        self.n_angles = torch.from_numpy(molecules[0].get_n_angles()).long()

        self.atom_types = self.particle_types

        # TODO add a warning or an error if "molecule" size > 1


    def wrap_positions(self, eps=1e-6):
        """
        Move atoms outside the box back into the box for all dimensions with periodic boundary
        conditions.

        Args:
            eps (float): Small offset for numerical stability
        """
        if torch.any(self.volume == 0.0):
            raise SystemWarning("Simulation cell required for wrapping of positions.")
        else:
            pbc_atomic = self.expand_atoms(self.pbc)

            # Compute fractional coordinates
            inverse_cell = torch.inverse(self.cells)
            inverse_cell = self.expand_atoms(inverse_cell)
            inv_positions = torch.sum(self.positions[..., None] * inverse_cell, dim=2)

            # Get periodic coordinates
            periodic = torch.masked_select(inv_positions, pbc_atomic)
            img = inv_positions // 1
            self.images = img.type(torch.long)

            # Apply periodic boundary conditions (with small buffer)
            periodic = periodic + eps
            periodic = periodic % 1.0
            periodic = periodic - eps

            # Update fractional coordinates
            inv_positions.masked_scatter_(pbc_atomic, periodic)

            # Convert to positions
            self.positions = torch.sum(
                inv_positions[..., None] * self.expand_atoms(self.cells), dim=2
            )