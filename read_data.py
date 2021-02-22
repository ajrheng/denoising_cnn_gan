"""
This file contains 
"""

import numpy as np
from PIL import Image

def get_mias_data(dir):
    """
    Returns a [BS x W x H x 1] array of MIAS images. BS = 322, W = 1024, H = 1024, 1 channel

    Args:
        dir: folder name containing the mammogram images
    """
    mias_data = []
    for i in range(1,301): # 300 training images
        filename = "dir" + "/mdb{:03d}.pgm".format(i) # in folder mias/
        data = Image.open(filename)
        data = np.array(data, dtype=np.uint8)
        mias_data.append(np.expand_dims(data, 2)) # make dimensions [1024, 1024, 1]

    mias_data = np.array(mias_data)
    return mias_data

def get_dx_data(dir):
    """
    Returns a [BS x W x H x 1] array of DX images. BS = 400, W = 1935, H = 2400, 1 channel

    Args:
        dir: folder name containing the dental images
    """
    dx_data = []
    for i in range(1,301): # 400 images in dx database
        filename = "dir" + "/{:03d}.bmp".format(i) # in folder dx/
        data = Image.open(filename)
        data = data.convert('L') # originally RGB so convert to grayscale
        data = np.array(data, dtype=np.uint8) # [1935, 2400]
        dx_data.append(np.expand_dims(data, 2)) # [1935, 2400, 1]
        
    dx_data = np.array(dx_data)
    return dx_data
