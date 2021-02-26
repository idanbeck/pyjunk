import torch
import torch.nn as nn

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
        )

    def forward(self, input, in_view):
        # broadcast view
        in_view = in_view.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        out = torch.cat((in_view, input), dim=1)
        representation = self.net.forward(out)

        return representation


