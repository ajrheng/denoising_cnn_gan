#!/usr/bin/env python

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
import torchvision.transforms
from torchvision.transforms import Compose
from torchvision.utils import save_image

IMGS_PATH = "amartel_data3/mmcneil/chest_xray/"
PYTORCH_TRANSFORM = Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.ToTensor()])

def conv_down(in_ch, out_ch):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=4, padding=1, stride=2)
    return layer

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.downsample = downsample

        if self.downsample:
            self.down_layer = conv_down(out_ch, out_ch)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.down_layer(out)
            return out_down, out
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, upass):
        upsampled = self.up(x)
        combined = torch.cat((upsampled, upass), 1)
        out = self.conv(combined)

        return out

class DeNoiser(nn.Module):
    def __init__(self, in_ch, frames=32, depth=2):
        super(DeNoiser, self).__init__()

        self.depth = depth
        self.collapse = nn.ModuleList()
        prev_channels = in_ch

        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.collapse.append(ConvBlock(prev_channels, frames*(2**i), downsample))
            prev_channels = frames * (2**i)

        self.restore = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.restore.append(UpBlock(prev_channels, frames*(2**i)))
            prev_channels = frames * (2**i)

        self.final_conv = nn.Conv2d(prev_channels, in_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        pass_forward = []
        for i, block in enumerate(self.collapse):
            if (i + 1) < self.depth:
                x, x_up = block(x)
                pass_forward.append(x_up)
            else:
                x = block(x)

        for i, block in enumerate(self.restore):
            x = block(x, pass_forward[-i-1])

        out = F.sigmoid(self.final_conv(x))
        return out
            


def training_loop(n_epochs, optimizer, model, loss_func, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            noisy = (torch.rand_like(imgs)/10).cuda() + imgs
            outputs = model(noisy)

            loss = F.mse_loss(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            save_image(noisy, str(epoch) + "_noisy.png")
            save_image(imgs, str(epoch) + "_real.png")
            save_image(outputs, str(epoch) + "_recon.png")

        with open("unet_loss.txt", "a") as resfile:
            print("Epoch {}: Training loss {}".format(
                epoch, float(loss_train)), file=resfile)

imgs = ImageFolder(IMGS_PATH, PYTORCH_TRANSFORM)
full = torch.utils.data.DataLoader(imgs, batch_size=128)

model = nn.DataParallel(DeNoiser(3)).cuda()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.NLLLoss().cuda().cuda()

training_loop(10, optimizer, model, loss_func, full)

torch.save(model.state_dict(), "tmp.pt")
