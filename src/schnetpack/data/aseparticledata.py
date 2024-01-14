import os
import torch
import logging
from schnetpack.data import ASEAtomsData 
from schnetpack.data.atoms import AtomsDataError, Atoms
from schnetpack.data.particles import Particles
from typing import List, Optional, Dict, Any
from schnetpack.data.sqlite_ext import SQLite3DatabaseExt
import schnetpack.properties as structure

logger = logging.getLogger(__name__)

class ASEParticlesData(ASEAtomsData):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database using ASE Particles.

    """

    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        """
        Args:
            datapath: Path to ASE DB.
            load_properties: Set of properties to be loaded and returned.
                If None, all properties in the ASE dB will be returned.
            load_properties: If True, load structure properties.
            transforms: preprocessing torch.nn.Module (see schnetpack.data.transforms)
            subset_idx: List of data indices.
            units: property-> unit string dictionary that overwrites the native units
                of the dataset. Units are converted automatically during loading.
        """
        
        super().__init__(
            datapath,
            load_properties=load_properties,
            load_structure=load_structure,
            transforms=transforms,
            subset_idx=subset_idx,
            property_units = property_units,
            distance_unit = distance_unit

        )

        self.conn = SQLite3DatabaseExt(datapath, create_indices=True, 
                               use_lock_file=True, serial=False)

    def _get_properties(self, conn, idx: int, load_properties: List[str], load_structure: bool ):
        properties = super()._get_properties(conn, idx, load_properties, load_structure)

        row = conn.get(idx + 1)

        if load_structure:
            properties[structure.particle_types] = (torch.tensor(row["particle_types"].copy()))
            properties[structure.molecule_ids] = (torch.tensor(row["molecule_ids"].copy()))
            properties[structure.bonds_list] = (torch.reshape(torch.tensor(row["bonds_list"].copy()),(-1,2)))
            properties[structure.angles_list] = (torch.reshape(torch.tensor(row["angles_list"].copy()),(-1,3)))
            properties[structure.bond_types] = (torch.tensor(row["bond_types"].copy()))
            properties[structure.angle_types] = (torch.tensor(row["angle_types"].copy()))
            properties[structure.n_bonds] = torch.tensor([len(properties[structure.bonds_list])])
            properties[structure.n_angles] = torch.tensor([len(properties[structure.angles_list])])

        return properties

    ## Creation

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEParticlesData":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
                reference values of the property. This is especially useful for
                extensive properties such as the energy, where the single atom energies
                contribute a major part to the overall value.
            kwargs: Pass arguments to init.

        Returns:
            newly created ASEParticlesData

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        atomrefs = atomrefs or {}

        #with connect(datapath) as conn:
        conn = SQLite3DatabaseExt(datapath, create_indices=True, 
                               use_lock_file=True, serial=False)
        #conn.columnnames = [line.split()[0].lstrip()
        #       for line in init_statements[0].splitlines()[1:]]
        conn.metadata = {
            "_property_unit_dict": property_unit_dict,
            "_distance_unit": distance_unit,
            "atomrefs": atomrefs,
        }
        
        return ASEParticlesData(datapath, **kwargs)


    def add_systems(
        self,
        property_list: List[Dict[str, Any]],
        atoms_list: Optional[List[Atoms]] = None,
    ):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list: System composition and geometry. If Atoms are None,
                the structure needs to be given as part of the property dicts
                (using structure.Z, structure.R, structure.cell, structure.pbc)
            property_list: Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset
                plus additional structure properties, if atoms is None.
        """
        if atoms_list is None:
            atoms_list = [None] * len(property_list)

        #with connect(datapath) as conn:
        conn = SQLite3DatabaseExt(self.datapath, create_indices=True, 
                               use_lock_file=True, serial=False)
        for at, prop in zip(atoms_list, property_list):
            self._add_system(conn, at, **prop)


    def _add_system(self, conn, atoms: Optional[Atoms] = None, **properties):
        """Add systems to DB"""
        if atoms is None:
            try:
                Z = properties[structure.Z]
                R = properties[structure.R]
                cell = properties[structure.cell]
                pbc = properties[structure.pbc]
                particle_types = properties[structure.particle_types]
                molecule_ids = properties[structure.molecule_ids]

                bonds_list =  properties[structure.bonds_list] 
                angles_list = properties[structure.angles_list]
                bond_types =  properties[structure.bond_types] 
                angle_types = properties[structure.angle_types]
            
                atoms = Particles(numbers=Z, positions=R, cell=cell, pbc=pbc, 
                    molecule_ids = molecule_ids, particle_types = particle_types,
                    bonds_list = bonds_list, angles_list = angles_list,
                    bond_types = bond_types, angle_types = angle_types)

            except KeyError as e:
                raise AtomsDataError(
                    "Property dict does not contain all necessary structure keys"
                ) from e

        # add available properties to database
        valid_props = set().union(
            conn.metadata["_property_unit_dict"].keys(),
            [
                structure.Z,
                structure.R,
                structure.cell,
                structure.pbc,
                structure.particle_types,
                structure.molecule_ids,
                structure.bonds_list, 
                structure.angles_list,
                structure.bond_types,
                structure.angle_types,
            ],
        )
        for prop in properties:
            if prop not in valid_props:
                logger.warning(
                    f"Property `{prop}` is not a defined property for this dataset and "
                    + f"will be ignored. If it should be included, it has to be "
                    + f"provided together with its unit when calling "
                    + f"AseAtomsData.create()."
                )

        data = {}
        for pname in conn.metadata["_property_unit_dict"].keys():
            try:
                data[pname] = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

        conn.write(atoms, data=data ) #, key_value_pairs = key_value_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.subset_idx is not None:
            idx = self.subset_idx[idx]

        props = self._get_properties(
            self.conn, idx, self.load_properties, self.load_structure
        ) #necessary because it used self.conn, so it took the one of the parent
        props = self._apply_transforms(props)

        return props

    # Metadata

    @property
    def metadata(self):
        #with connect(datapath) as conn:
        conn = SQLite3DatabaseExt(self.datapath, create_indices=True, 
                               use_lock_file=True, serial=False)
        return conn.metadata

    def _set_metadata(self, val: Dict[str, Any]):
        #with connect(datapath) as conn:
        conn = SQLite3DatabaseExt(self.datapath, create_indices=True, 
                               use_lock_file=True, serial=False)
        conn.metadata = val