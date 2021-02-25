"""
This file contains class definitions of Datasets that you can use with Pytorch's Dataloader.
"""

import numpy as np
from torch.utils.data import Dataset
import torch

class mias_dataset(Dataset):
    """
    Dataset class for MIAS (mammogram) data
    """
    def __init__(self, clean_mias_data, clean_transform, noise_transform):
        """
        Args:
            clean_transform: composition of transforms for the clean data
            noise_transform: composition of transforms for the noisy data
        """
        self.clean_data = torch.clone(clean_mias_data)
        self.noise_data = torch.clone(self.clean_data)
        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, index):
        return self.clean_transform(self.clean_data[index]), self.noise_transform(self.noise_data[index])
    
class dx_dataset(Dataset):
    """
    Dataset class for DX (dental) data
    """
    def __init__(self, clean_dx_data, clean_transform, noise_transform):
        """
        Args:
            clean_transform: composition of transforms for the clean data
            noise_transform: composition of transforms for the noisy data
        """
        self.clean_data = torch.clone(clean_dx_data)
        self.noise_data = torch.clone(self.clean_data)
        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, index):
        return self.clean_transform(self.clean_data[index]), self.noise_transform(self.noise_data[index])

