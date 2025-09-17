# src/models/attention.py
#minimal spectral-spatial attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralSpatialAttention(nn.Module):
    """
    Lightweight spectral-spatial attention used in generator.
    Keeps it small for easy debugging.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        s = x.mean(dim=(2,3), keepdim=True)   # channel descriptor
        s = self.conv1(s)
        s = F.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        return x * s
