from typing import Dict

import torch
import torch.nn as nn
import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = ["ExcludedVolumeEnergy"]

class ExcludedVolumeEnergy(nn.Module):
    """Class for the calculation of excluded volume energy on a selected list of particles
    """
    def __init__(self, sigma: float, 
                 exponent: int, 
                 output_key: str = "excluded_volume_energy", 
                 filter_key: str = "complete"):
        """
        Args:
            sigma: reference distance in the same units as "distance_unit" used in the dataset
            exponent: exponent used in the calculation:

                excluded_volume_energy = (sigma/R_ij)^exponent
            
            output_key: the key under which the result will be stored
            filter_key: selector of the subset of the neighbors list over which the calculation
                will be performed. Accepted values:
                - complete (default): perform the calculation considering all the entries of the
                                      neighbors list
                - bonded: perform the calculation only for the particles listed in `inputs[properties.bonds_list]`
                - nonbonded: perform the calculation only for the particles not listed in `inputs[properties.bonds_list]`
                - intermolecular: perform the calculation only for pairs of particles belonging to different molecules
                - intramolecular: perform the calculation only for pairs of particles belonging to the same molecule
                
        """
        super(ExcludedVolumeEnergy, self).__init__()
        self.sigma = sigma
        self.exponent = exponent
        self.output_key = output_key
        self.filter_key = filter_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Extract the device 
        device = inputs[properties.idx_i].device

        # Select the particle pairs list to include in the calculation
        if self.filter_key == "complete":
            distances = torch.norm(inputs[properties.Rij], dim=1)
        elif self.filter_key == "bonded":
            distances = torch.norm(inputs[properties.Rij][inputs[properties.idx_of_bonds],:], dim=1)
        elif self.filter_key == "nonbonded":
            distances = torch.norm(inputs[properties.Rij][inputs[properties.idx_non_bonds],:], dim=1)
        elif self.filter_key == "intermolecular":
            distances = torch.norm(inputs[properties.Rij][inputs[properties.idx_of_inter],:], dim=1)
        elif self.filter_key == "intramolecular":
            distances = torch.norm(inputs[properties.Rij][inputs[properties.idx_of_intra],:], dim=1)
        else:
            raise NotImplementedError("The requested filter_key does not correspond to any available implementation.")

        # Compute the per-pair excluded volume energy
        # Note: periodic image flags are already considered in Rij by PairwiseDistances
        excluded_volume_energy_per_pair = (distances.reciprocal() * self.sigma).pow(self.exponent).to(device)

        if excluded_volume_energy_per_pair.numel() > 0:
            #If the neighbors list was not empty, aggregate results per frame
            excluded_volume_energy = self.aggregate_by_frame(inputs, excluded_volume_energy_per_pair)
        else:
            # If the neighbours list was empty, return 0s
            n_frames = inputs[properties.n_atoms].size()[0]
            excluded_volume_energy = torch.zeros(n_frames, device=device)

        # Dividing by half because each atom pair appears twice in the distances list
        excluded_volume_energy = excluded_volume_energy * 0.5 

        inputs[self.output_key] = excluded_volume_energy  
        
        return inputs

    def aggregate_by_frame(self, inputs: Dict[str, torch.Tensor], 
                           excluded_volume_energy_per_pair: torch.Tensor) -> torch.Tensor:
            # The number of neighbours is different in every frame, therefore also the 
            # number of contributions to the total excluded volume energy is different
            # in each frame. It is necessary to retrieve the frame to which each atom
            # pair belongs, in order to aggregate the predictions correctly.
            # We exploit the fact that idx_i are sorted ascending and their id numbering 
            # is ultiplied by iframe+1. E.g. for 500 atoms:
            #    atom 0 in frame 0 has idx 0, but in frame 1 has idx 500
            
            n_atoms = inputs[properties.n_atoms]
            n_frames = inputs[properties.n_atoms].size()[0]

            # Select the particle pairs list to include in the calculation
            if self.filter_key == "complete":
                idx_i = inputs[properties.idx_i]
            elif self.filter_key == "bonded":
                idx_i = inputs[properties.idx_i][inputs[properties.idx_of_bonds]]
            elif self.filter_key == "nonbonded":
                idx_i = inputs[properties.idx_i][inputs[properties.idx_non_bonds]]
            elif self.filter_key == "intermolecular":
                idx_i = inputs[properties.idx_i][inputs[properties.idx_of_inter]]
            elif self.filter_key == "intramolecular":
                idx_i = inputs[properties.idx_i][inputs[properties.idx_of_intra]]
            else:
                raise NotImplementedError("The requested filter_key does not correspond to any available implementation.")

            device = inputs[properties.idx_i].device
            
            # List of the maximum atom index in each frame, prepended with 0
            # e.g. : for 3 frames with 500 atoms each we end up with tensor([0, 500, 1000, 1500])
            cumulative_n_atoms = torch.tensor([n_atoms[0]])
            for i in range(1, n_frames):
                next_max_idx = torch.tensor([n_atoms[i] + cumulative_n_atoms[i-1]])
                cumulative_n_atoms = torch.cat((cumulative_n_atoms, next_max_idx), dim = 0)

            cumulative_n_atoms = torch.cat((torch.tensor([0]), cumulative_n_atoms) , dim = 0).detach()            
            
            # List of the frame indices of each pair
            idx_frame = torch.zeros(idx_i.size()[0])
            idx_frame = idx_frame.int()
            for iframe in range(n_frames):
                inf_lim = cumulative_n_atoms[iframe]
                sup_lim = cumulative_n_atoms[iframe+1]
                idx_for_this_frame = torch.where((idx_i>=inf_lim) & (idx_i<sup_lim))
                # [0] in the next line used because torch.where returns a tuple of size 1, not a tensor
                idx_frame[idx_for_this_frame[0]] = iframe
            
            # Aggregate excluded_volume_energy for each frame
            max_idf = int(idx_frame[-1]) + 1
            excluded_volume_energy = snn.scatter_add(excluded_volume_energy_per_pair, idx_frame.to(device).detach(), dim_size = max_idf)
            excluded_volume_energy = torch.squeeze(excluded_volume_energy, -1)

            return excluded_volume_energy
