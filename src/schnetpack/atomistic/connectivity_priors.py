from typing import List, Dict, Union

import torch
import torch.nn as nn
import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = ["HarmonicBondPrior"]
        
class HarmonicBondPrior(nn.Module):
    """Class for the calculation of bond energy using an harmonic function.
    """
    def __init__(self, stiffness: List[Union[int, float]], 
                       equilibrium_value: List[Union[int, float]], 
                       output_key: str = "bond_energy"):
        """
        Args:
            stiffness: list of ints of floats. It must contain an entry for each bond type present in the system. 
                It represens the spring stiffness in the harmonic function used to represent the bond energy.
            equilibrium_value: list of ints of floats. It must contain an entry for each bond type present in the system
                It represens the equilibrium value of the bond length, used in the harmonic function used to represent 
                the bond energy:

                bond_energy = stiffness * (distance - equilibrium_value)^2
            
            output_key: the key under which the result will be stored
          
        """
        
        super(HarmonicBondPrior, self).__init__()

        self.stiffness = torch.tensor(stiffness) # Conversion for compatibility with Torchscript
        self.equilibrium_value = torch.tensor(equilibrium_value) # Conversion for compatibility with Torchscript
        self.output_key = output_key
        self.model_outputs = [output_key]

        self.initialize_lists = True  # It will instructs to construct the stiffness_list and equilibrium_values_list 
                                      # at the first forward pass
        # Lists of len = n_bonds * batch_size for convenience in the elementwise multiplication with the distances
        self.stiffness_list = torch.empty(1) 
        self.equilibrium_values_list = torch.empty(1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract the device used
        device = inputs[properties.idx_i].device

        # Initialize the lists according to the batch size
        if self.initialize_lists:
            self.initialize_lists = False
            self.initialize(inputs)

        # If the batch_size has changed from the previous pass, re-compute stiffness_list and equilibrium_values_list
        if len(self.stiffness_list) != len(inputs[properties.bonds_list]):
            self.reset(inputs)
        
        # Extract the distance vectors corresponding to the bonded pairs
        Rij_bonds = inputs[properties.Rij][inputs[properties.idx_of_bonds],:]
        # Compute the distance value
        bond_distances = torch.norm(Rij_bonds, dim=1)

        # Comupte the bonded energy contribution per-pair
        E_bond_batch = self.stiffness_list * (bond_distances - self.equilibrium_values_list).pow(2).to(device)

        # Sum the values for the bonds included in the same frame
        E_bond = self.aggregate_by_frame(inputs, E_bond_batch)

        # Save the result in inputs
        inputs[self.output_key] = E_bond * 0.5 # because the bonds list contains every bond twise, e.g 0-1 and 1-0

        return inputs

    def initialize(self, inputs: Dict[str, torch.Tensor]) -> None:
        # Create a list of the stiffness and equilibrium_value values for each bond in the batch, according to their type
        bond_types = inputs[properties.bond_types]
        # Doubling because the list of neighbors contains both entries for a bond, e.g. [0, 1] and [1, 0]
        bond_types = torch.cat((bond_types, bond_types), 0)

        device = inputs[properties.idx_i].device

        stiffness = self.stiffness.to(device).detach()
        self.stiffness_list = stiffness[bond_types.long().to(device)]
        
        equilibrium_value = self.equilibrium_value.to(device).detach()
        self.equilibrium_values_list = equilibrium_value[bond_types.long().to(device)]

    def reset(self, inputs: Dict[str, torch.Tensor]) -> None:
        # Reset the values and call the initialization with the new input to adjust the size of the lists
        self.stiffness_list = torch.empty(1)
        self.equilibrium_values_list = torch.empty(1)
        self.initialize(inputs)

    def aggregate_by_frame(self, inputs: Dict[str, torch.Tensor], 
                           E_bond_batch: torch.Tensor) -> torch.Tensor:
        # Compute the total bond energy for each frame by summing the values corresponding to 
        # the bonds present in each frame.

        # List of the total number of bonds present in each frame
        n_bonds = inputs[properties.n_bonds] * 2
        # n_bonds had the same number of elements as the batch size
        batch_size = len(n_bonds)
        # Extract the device used
        device = inputs[properties.idx_i].device

        # Create a list of the frame index which corresponds to each E_bond_batch entry.
        # We use the number of bonds in each frame n_bonds[i] to know how many times to enter the 
        # corresponding frame number i.
        idx_frame_list = [[i]*n_bonds[i].item() for i in range(batch_size)]
        idx_frame_list_tensor = torch.tensor(idx_frame_list)
        idx_frame = torch.flatten(idx_frame_list_tensor)  #flatten

        # Using the frame index list and the batch size, we aggregate the elements of E_bond_batch
        # that have the same frame index, using scatter_add.
        E_bond = snn.scatter_add(E_bond_batch, idx_frame.to(device).detach(), dim_size = batch_size)
        E_bond = torch.squeeze(E_bond, -1)

        return E_bond
