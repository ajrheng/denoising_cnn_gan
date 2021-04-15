from skimage.util import random_noise
import torch
import numpy as np

class GaussianNoise():
    """
    Torchvision transform to addd Gaussian noise. 

    Use it as you would a regular torchvision transform.
    """
    def __init__(self, mean=0, var_min=0.005, var_max = 0.05):
        """

        Args:
            mean: mean of the Gaussian noise
            var: variance of Gaussian noise
        """
        self.mean = mean
        self.var_min = var_min
        self.var_max = var_max
        
    def __call__(self, tensor):
        var = np.random.uniform(low=self.var_min, high=self.var_max)
        return torch.tensor(random_noise(tensor, mode='gaussian', mean=self.mean, var=var))
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)