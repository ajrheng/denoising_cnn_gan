import torch.nn as nn
import torch.nn.functional as F
import torch

class cnn_autoencoder(nn.Module):
    
    def __init__(self):
        super(cnn_autoencoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1) # [1, 64, 64] -> [64, 64, 64]
        self.pool1 = nn.MaxPool2d(2) # [64, 64, 64] -> [64, 32, 32]
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1) # [64, 32, 32] -> [64, 32, 32]
        self.pool2 = nn.MaxPool2d(2) # [64, 32, 32] -> [64, 16, 16]
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1) # [64, 16, 16] -> [64, 16, 16]
        self.upsamp4 = nn.Upsample(scale_factor=2) # [64, 16, 16] -> [64, 32, 32]
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1) # [64, 32, 32] -> [64, 32, 32]
        self.upsamp5 = nn.Upsample(scale_factor=2) # [64, 32, 32] -> [64, 64, 64]
        self.conv5 = nn.Conv2d(64, 1, 3, padding=1) # [64, 64, 64] -> [1, 64, 64]
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.upsamp4(x)
        x = F.relu(self.conv4(x))
        x = self.upsamp5(x)
        x = torch.sigmoid(self.conv5(x)) # convert all outputs to range [0,1]
        return x