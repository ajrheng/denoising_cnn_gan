import numpy as np
from PIL import Image

def get_mias_data(dir, test=False):
    """
    Returns a [BS x W x H x 1] array of MIAS images. BS = 300 if train, 22 if test, W = 1024, H = 1024, 1 channel

    Args:
        dir: folder name containing the mammogram images
        test: boolean to toggle between loading train or test data
    """
    mias_data = []

    if test:
        low_range = 301
        upp_range = 323
    else:
        low_range = 1
        upp_range = 301

    for i in range(low_range,upp_range): 
        filename = dir + "/mdb{:03d}.pgm".format(i) # in folder mias/
        data = Image.open(filename)
        data = np.array(data, dtype=np.uint8)
        mias_data.append(np.expand_dims(data, 2)) # make dimensions [1024, 1024, 1]

    mias_data = np.array(mias_data)
    return mias_data

def get_dx_data(dir, test=False):
    """
    Returns a [BS x W x H x 1] array of DX images. BS = 300 if train, 100 if test, W = 1935, H = 2400, 1 channel

    Args:
        dir: folder name containing the dental images
        test: boolean to toggle between loading train or test data
    """
    if test:
        low_range = 301
        upp_range = 401
    else:
        low_range = 1
        upp_range = 301
    
    dx_data = []
    for i in range(low_range,upp_range): 
        filename = dir + "/{:03d}.bmp".format(i) # in folder dx/
        data = Image.open(filename)
        data = data.convert('L') # originally RGB so convert to grayscale
        data = np.array(data, dtype=np.uint8) # [1935, 2400]
        dx_data.append(np.expand_dims(data, 2)) # [1935, 2400, 1]
        
    dx_data = np.array(dx_data)
    return dx_data
