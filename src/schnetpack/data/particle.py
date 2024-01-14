from ase.atom import Atom, names
import numpy as np

"""Definition of the Particle classe, which extend the ASE  
Atom with typing and molecule membership, to generealize the 
description of particle-based bulk systems made of 
Coarse Grained moieties.
"""
# Update names with new Particle properties
# Singular, plural, default value:
names['particle_type'] = ('particle_types', None)
names['molecule_id'] =   ('molecule_ids', None)


class Particle(Atom):
    """Class for representing a single particle, subclassing ASE Atom.
    The get method overrides the behaviour of the parent class: it will 
    return an error if an attribute was not explicitly set.

    Parameters added to Particle:

    type: int
        Particle type
    molecule_id: int
        Id of the molecule the particle belongs to
    """    
    def __init__(self, symbol='X', position=(0, 0, 0),
                 tag=None, momentum=None, mass=None,
                 magmom=None, charge=None,
                 atoms=None, index=None,

                 particle_type=0, molecule_id=0):

        super().__init__(symbol, position,
                 tag, momentum, mass, magmom, charge,
                 atoms, index)

        self.data['particle_type'] = particle_type
        self.data['molecule_id'] = molecule_id

    def __repr__(self):
        s = super().__repr__() 
        s = s + " Particle ("
        for name in ['particle_type', 'molecule_id']:
            value = self.get_raw(name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                s += ', %s=%s' % (name, value)
        if self.atoms is None:
            s += ')'
        else:
            s += ', index=%d)' % self.index
        return s

    def get(self, name):
        """Get name attribute, return an error if not explicitly set."""
        value = self.get_raw(name)
        # get_raw gets name attribute, returns None if not explicitly set
        if value is None:
            raise RuntimeError("The value of " + name + " was not defined")
        return value