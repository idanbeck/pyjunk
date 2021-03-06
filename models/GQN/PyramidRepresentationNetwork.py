import torch
import torch.nn as nn

import repos.pyjunk.junktools.pytorch_utils  as ptu

# Pyramid Representation Network

class PyramidRepresentationNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PyramidRepresentationNetwork, self).__init__(*args, **kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(3 + 7, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU(),
        ).to(ptu.GetDevice())

    def forward(self, input, in_view):
        # broadcast view
        in_view = in_view.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)

        # print(in_view.shape)
        # print(input.shape)

        out = torch.cat((in_view, input), dim=1)
        representation = self.net.forward(out)

        #print(representation.shape)

        return representation


