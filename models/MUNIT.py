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

        # input layer
        self.net.extend([
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7)
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        ])

        # Downsampling
        for i in range(self.n_downsample):
            self.net.extend([
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2),
                ),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ])
            channels *= 2

        # ResBlocks
        for n in range(self.n_res_blocks):
            self.net.append(
                ResBlock(n_channels=channels)
            )

        self.net = nn.Sequential(*self.net)
        self.out_channels = channels

    def forward(self, in_x):
        return self.net(in_x)

    @property
    def channels(self):
        return self.out_channels

class MUNITStyleEncoder(nn.Module):
    def __init__(self, n_channels=64, n_downsample=4, style_dim=8, *args, **kwargs):
        super(MUNITStyleEncoder, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.n_downsample = n_downsample
        self.style_dim = style_dim
        self.n_deepen_layers = 2
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []
        channels = self.n_channels

        # input layer
        self.net.extend([
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7, padding=0)
            ),
            nn.ReLU(inplace=True)
        ])

        # downsampling layers
        for i in range(self.n_deepen_layers):
            self.net.extend([
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True)
            ])
            channels *= 2

        for i in range(self.n_downsample - self.n_deepen_layers):
            self.net.extend([
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True)
            ])

        # global pooling and pointwise convolution to style channels
        self.net.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.style_dim, kernel_size=1)
        ])

        self.net = nn.Sequential(*self.net)
        self.out_channels = channels

    def forward(self, in_x):
        return self.net(in_x)


class MUNIT(Model):
    def __init__(self, *args, **kwargs):
        super(MUNIT, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        pass