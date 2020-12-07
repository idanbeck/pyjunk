import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.models.modules.MLPBlock import MLPBlock

# Simple MLP model

class SimpleMLP(Model):
    def __init__(self, dim_input, dim_output, dim_inner, n_blocks, n_layers, fResidual=False, *args, **kwargs):
        super(SimpleMLP, self).__init__(*args, **kwargs)
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_inner = dim_inner
        self.fResidual = fResidual

        self.net = []
        self.ConstructModel()

    def ConstructModel(self):
        self.net = []

        for block in range(self.n_blocks):
            self.net.extend([
                MLPBlock(
                    self.dim_input,
                    self.dim_output,
                    [self.dim_inner] * self.n_layers,
                    fResidual=self.fResidual
                ),
                nn.ReLU()   # TODO: Generalize this shit
            ])

        # eliminate the last baddie
        self.net.pop()
        self.net = nn.ModuleList([*self.net])

    def forward(self, torchInput):
        batch_size = 1

        # Flatten for MLP
        H, W, C = torchInput.shape
        out = torchInput.reshape(batch_size, H*W*C)

        for block in self.net:
            out = block(out)

        # Reshape back into image
        out = out.reshape(H, W, C)

        return out

    # TODO: This is not very general
    def loss(self, torchInput):
        # Ultimately we just want to determine an L2 reconstruction loss
        out = self.forward(torchInput)
        l2_reconstruction_loss = nn.MSELoss()
        out = l2_reconstruction_loss(torchInput, out)
        return out