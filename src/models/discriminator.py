# src/models/discriminator.py
#small patch discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSIGeneDiscriminator(nn.Module):
    def __init__(self, in_channels=200):
        super().__init__()
        c = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*2, c*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*4, 1, 4, padding=1)
        )
    def forward(self, x):
        return self.net(x).mean(dim=[1,2,3])  # return a single value per batch sample
