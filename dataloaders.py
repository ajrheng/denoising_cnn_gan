"""
This file contains class definitions of Datasets that you can use with Pytorch's Dataloader.
"""

from read_data import get_mias_data, get_dx_data
import numpy as np
from torch.utils.data import Dataset

class mias_dataset(Dataset):
    """
    Dataset class for MIAS (mammogram) data
    """
    def __init__(self, dir, clean_transform, noise_transform):
        """
        Args:
            clean_transform: composition of transforms for the clean data
            noise_transform: composition of transforms for the noisy data
        """
        self.clean_data = get_mias_data(dir)
        self.noise_data = np.copy(self.clean_data)
        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, index):
        return clean_transform(self.clean_data[index]), noise_transform(self.noise_data[index])
    
class dx_dataset(Dataset):
    """
    Dataset class for DX (dental) data
    """
    def __init__(self, dir, clean_transform, noise_transform):
        """
        Args:
            clean_transform: composition of transforms for the clean data
            noise_transform: composition of transforms for the noisy data
        """
        self.clean_data = get_dx_data(dir)
        self.noise_data = np.copy(self.clean_data)
        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, index):
        return clean_transform(self.clean_data[index]), noise_transform(self.noise_data[index])

