import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.models.modules.ResnetBlock import ResnetBlock

# Simple Resnet model

class SimpleResnet(Model):
    def __init__(self, n_blocks, *args, **kwargs):
        super(SimpleResnet, self).__init__(*args, **kwargs)
        self.n_blocks = n_blocks

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []

        for block in range(self.n_blocks):
            self.net.extend([
                ResnetBlock(self.dim_input, self.dim_output, [self.dim_inner] * self.n_layers, fResidual=True),
                nn.ReLU()  # TODO: Generalize this shit
            ])

        # eliminate the last baddie
        self.net.pop()
        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for block in self.net:
            out = block(out)
        return out