import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, n_filters, *args, inner_kernel=3, **kwargs):
        super(ResnetBlock, self).__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.inner_kernel = inner_kernel

        # Construct the network
        self.net = [
            nn.Conv2d(self.n_filters, self.n_filters, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters, self.inner_kernel, stride=1, padding=self.inner_kernel // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters, 1, stride=1, padding=0),
            nn.ReLU(),
        ]
        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)
        return out

