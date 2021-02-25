This folder contains code for processing raw MIAS and DX data into datasets for denoising.
The datasets are stored as torch tensors of sizes [BS, 1, 128, 128], where BS is the batch size.
The images are grayscale (1 in second dimension) and resized to 128 x 128 pixels.

MIAS dataset: http://peipa.essex.ac.uk/info/mias.html
DX dataset: http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/

## MIAS dataset

Contains 322 images of mammograms. This is split into 300 images for the train set and 22 images for test set.

## DX dataset

Contains 400 images of dental images. Split into 300 images for the train set and 100 images for test set.