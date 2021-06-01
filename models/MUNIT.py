import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
from repos.pyjunk.models.modules.AdaptiveInstanceNorm2d import AdaptiveInstanceNorm2d

import math

# MUNIT

class ResBlock(nn.Module):
    def __init__(self, n_channels, style_dim=None, hidden_dim=None, *args, **kwargs):
        super(ResBlock, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3)
            ),
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3)
            )
        )

        self.use_style = self.style_dim is not None and self.hidden_dim is not None
        if(self.use_style):
            self.norm1 = AdaptiveInstanceNorm2d(self.n_channels, self.style_dim, self.hidden_dim)
            self.norm2 = AdaptiveInstanceNorm2d(self.n_channels, self.style_dim, self.hidden_dim)
        else:
            self.norm1 = nn.InstanceNorm2d(self.n_channels)
            self.norm2 = nn.InstanceNorm2d(self.n_channels)

        self.activation = nn.ReLU()

    def forward(self, in_x, in_style=None):
        out = in_x
        out = self.conv1(out)
        out = self.norm1(out, in_style) if self.use_style else self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out, in_style) if self.use_style else self.norm1(out)
        return out + in_x

class MUNITContentEncoder(nn.Module):
    def __init__(self, n_channels=64, n_downsample=2, n_res_blocks=4, *args, **kwargs):
        super(MUNITContentEncoder, self, *args, **kwargs)
        self.n_channels = n_channels
        self.n_downsample = n_downsample
        self.n_res_blocks = n_res_blocks
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []
        channels = self.n_channels
        self.net.extend([
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7)
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        ])

        for i in range(self.n_downsample):




class MUNIT(Model):
    def __init__(self, *args, **kwargs):
        super(MUNIT, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        pass