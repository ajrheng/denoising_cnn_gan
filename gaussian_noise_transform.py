from skimage.util import random_noise

class GaussianNoise():
    """
    Torchvision transform to addd Gaussian noise. 

    Use it as you would a regular torchvision transform.
    """
    def __init__(self, mean=0., var=0.01):
        """

        Args:
            mean: mean of the Gaussian noise
            var: variance of Gaussian noise
        """
        self.mean = mean
        self.var = var
        
    def __call__(self, tensor):
        return random_noise(tensor, mode='gaussian', mean=self.mean, var=self.var)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)