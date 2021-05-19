import torch
import torch.nn as nn
import torch.nn.Functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
import torch.optim as optim
import repos.pyjunk.junktools.pytorch_utils  as ptu

from repos.pyjunk.models.modules.VGGLoss import VGGLoss

# Super Resolution GAN

# ResBlock using PReLU
class ResBlock(nn.Module):
    def __init__(self, n_channels, *args, **kwargs):
        super(ResBlock, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        self.n_layers = 2

        self.net = []

        for l in range(self.n_layers):
            self.net.extend([
                nn.Conv2d(self.n_channels,
                          self.n_channels,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding),
                nn.BatchNorm2d(self.n_channels),
                nn.PReLU(),
            ])

        self.net.pop()
        self.net = nn.ModuleList([*self.net])

    def forward(self, in_x):
        out = in_x
        for layer in self.net:
            out = layer.forward(out)
        return out + in_x

# Pixel Shuffle Block
class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor=4, *args, **kwargs):
        super(PixelShuffleBlock, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_factor = up_factor

        self.net = []

        self.net.append([
            nn.Conv2d(self.in_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PixelShuffle(self.up_factor),
            nn.PReLU()
        ])

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer.forward(out)
        return out

class SRGAN_Generator(nn.Module):
    def __init__(self, n_channels=64, n_resblocks=16, n_ps_blocks=2, *args, **kwargs):
        super(SRGAN_Generator, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.n_resblocks = n_resblocks
        self.n_ps_blocks = n_ps_blocks
        self.ConstructModel()

    def ConstructModel(self):

        C = 3  # TODO: generalize

        # input layer
        self.input_layer = nn.ModuleList([
            nn.Conv2d(C, self.n_channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        ])

        # ResBlocks
        self.res_layer = []
        for r_l in range(self.n_resblocks):
            self.res_layer.append(
                ResBlock(n_channels = self.n_channels)
            )
        self.res_layer = nn.ModuleList([*self.res_layer])

        # Pixel Shuffle Blocks
        self.n_ps_layer = []
        channel_count = self.n_channels
        mult_factor = 4
        up_factor = 2
        for p_l in range(self.n_ps_blocks):
            self.n_ps_layer.append(
                PixelShuffleBlock(
                    in_channels=channel_count,
                    out_channels=channel_count * mult_factor,
                    up_factor=up_factor
                )
            )
            channel_count *= mult_factor
        self.n_ps_layer = nn.ModuleList([*self.n_ps_layer])

        # output layer
        self.out_layer = nn.ModuleList([
            nn.Conv2d(self.n_channels, C, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        ])

    def forward(self, input):
        out = input

        out_input_layer = self.input_layer.forward(out)
        out_res_layer = self.res_layer(out_input_layer) + out_input_layer
        out_ps_layer = self.ps_layer(out_res_layer)
        out = self.out_layer(out_ps_layer)

        return out

class SRGAN_Discriminator(nn.Module):
    def __init__(self, n_channels=64, n_blocks=3, *aegs, **kwargs):
        super(SRGAN_Discriminator, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.n_blocks = n_blocks

        self.net = []

        C = 3

        # input layer
        self.net.extend([
            nn.Conv2d(C, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=2, padding=1)
            nn.BatchNorm2d(self.n_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # res blocks
        channel_count = self.n_channels
        factor = 2
        for b in range(self.n_blocks):
            self.net.extend([
                nn.Conv2d(channel_count, channel_count * factor, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2(channel_count * factor),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(channel_count, channel_count * factor, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2(channel_count * factor),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            channel_count *= 2

        # output block
        self.net.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel_count, channel_count * factor, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_count * factor, 1, kernel_size=1, stride=1, padding=0),
            nn.Flatten()  # sigmoid can be added in loss fn
        ])

        self.net = nn.ModlueList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer.forward(out)
        return out


class SRGAN(Model):
    def __init__(self, input_shape, *args, **kwargs):
        self.input_shape = input_shape
        super(SRGAN, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        C, H, W = self.input_shape
        img_dim = C * H * W

        self.generator = SRGAN_Generator(
            n_channels=64,
            n_resblocks=16,
            n_ps_blocks=2
        ).to(ptu.GetDevice())

        self.discriminator = SRGAN_Discriminator(
            n_channels=64,
            n_blocks=3
        ).to(ptu.GetDevice())

        self.vgg_loss = VGGLoss(
            requires_grad=False
        ).to(ptu.GetDevice())



