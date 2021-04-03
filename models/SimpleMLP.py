import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.models.modules.MLPBlock import MLPBlock

import repos.pyjunk.junktools.pytorch_utils as ptu

# Simple MLP model

class SimpleMLP(Model):
    def __init__(self, dim_input, dim_output, dim_inner, n_blocks, n_layers, fResidual=False, *args, **kwargs):
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_inner = dim_inner
        self.fResidual = fResidual
        super(SimpleMLP, self).__init__(*args, **kwargs)


    def ConstructModel(self):
        self.net = []

        for block in range(self.n_blocks):
            self.net.extend([
                # Block should output the same dim as the output of the last block if not input block
                # TODO: Generalize this more
                MLPBlock(
                    self.dim_input if block == 0 else self.dim_output,
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
        # Flatten for MLP
        batch_size, H, W, C = torchInput.shape
        out = torchInput.reshape(batch_size, H * W * C)

        for block in self.net:
            out = block(out)

        # Reshape back into image
        # TODO: Generalize this  (force 3 channel output right now)
        #out = out.reshape(H, W, C)
        out = out.reshape(batch_size, H, W, 3)

        return out

    # TODO: This is not very general
    def loss(self, torchInput):
        # Ultimately we just want to determine an L2 reconstruction loss
        out = self.forward(torchInput)

        #l2_reconstruction_loss = nn.MSELoss()
        #loss = l2_reconstruction_loss(torchInput, out)

        l1_reconstruction_loss = nn.L1Loss()
        loss = l1_reconstruction_loss(torchInput, out)

        l2_lambda = 0.01
        #l2_reg = torch.tensor(0.)
        l2_reg = ptu.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        return loss

    def loss_with_target(self, torchSourceImageBuffer, torchTargetImageBuffer):
        out = self.forward(torchSourceImageBuffer)

        # l2_reconstruction_loss = nn.MSELoss()
        # loss = l2_reconstruction_loss(out, torchTargetImageBuffer)

        l1_reconstruction_loss = nn.L1Loss()
        loss = l1_reconstruction_loss(torchTargetImageBuffer, out)

        l2_lambda = 0.01
        # l2_reg = torch.tensor(0.)
        l2_reg = ptu.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        return loss