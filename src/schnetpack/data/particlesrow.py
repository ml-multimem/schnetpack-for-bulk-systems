from ase.db.row import AtomsRow
from schnetpack.data.particles import Particles
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import get_calculator_class, all_properties


class ParticlesRow(AtomsRow):
    def __init__(self, particles):
        super().__init__(particles)

        if isinstance(particles,dict):
            self.particle_types = particles["particle_types"].copy()
            self.molecule_ids = particles["molecule_ids"].copy()
            self.bonds_list = particles["bonds_list"].copy()
            self.angles_list = particles["angles_list"].copy()
            self.bond_types = particles["bond_types"].copy()
            self.angle_types = particles["angle_types"].copy()
        elif isinstance(particles, Particles):
            self.particle_types = particles.get_particle_types()
            self.molecule_ids = particles.get_molecule_ids()
            self.bonds_list = particles.get_bonds_list()
            self.angles_list = particles.get_angles_list()
            self.bond_types = particles.get_bond_types()
            self.angle_types = particles.get_angle_types()
        else:
            raise RuntimeError("Get method for this data type not implemented")

    def toatoms(self, attach_calculator=False,
                add_additional_information=False):
        """Create Particles object."""
        atoms = Particles(self.numbers,
                      self.positions,
                      cell=self.cell,
                      pbc=self.pbc,
                      magmoms=self.get('initial_magmoms'),
                      charges=self.get('initial_charges'),
                      tags=self.get('tags'),
                      masses=self.get('masses'),
                      momenta=self.get('momenta'),
                      constraint=self.constraints, 
                      particle_types = self.particle_types, 
                      molecule_ids = self.molecule_ids,
                      bonds_list = self.bonds_list, 
                      angles_list = self.angles_list,
                      bond_types = self.bond_types, 
                      angle_types = self.angle_types)

        if attach_calculator:
            params = self.get('calculator_parameters', {})
            atoms.calc = get_calculator_class(self.calculator)(**params)
        else:
            results = {}
            for prop in all_properties:
                if prop in self:
                    results[prop] = self[prop]
            if results:
                atoms.calc = SinglePointCalculator(atoms, **results)
                atoms.calc.name = self.get('calculator', 'unknown')

        if add_additional_information:
            atoms.info = {}
            atoms.info['unique_id'] = self.unique_id
            if self._keys:
                atoms.info['key_value_pairs'] = self.key_value_pairs
            data = self.get('data')
            if data:
                atoms.info['data'] = data

        return atoms