import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.models.modules.ResnetBlock import ResnetBlock

# simple resnet model

class SimpleResnet(Model):
    def __init__(self, input_shape, output_shape, n_blocks=4,
                 n_filters=64, n_layers=4, kernel_size=7, fResidual=True,
                 *args, **kwargs):
        super(SimpleResnet, self).__init__(*args, **kwargs)
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.fResidual = fResidual

        self.net = []
        self.ConstructModel()

    def ConstructModel(self):
        self.net = []

        for block in range(self.n_blocks):
            self.net.extend([
                ResnetBlock(
                    n_filters=self.n_filters,
                    n_layers=self.n_layers,
                    kernel_size=self.kernel_size,
                    fResidual=self.fResidual),
                nn.ReLU()  # TODO: moar generalize
            ])

        # eliminate the last baddie
        self.net.pop()
        self.net = nn.modulelist([*self.net])

    def forward(self, input):
        out = input
        for block in self.net:
            out = block(out)
        return out

    def loss(self, torchInput):
        # Ultimately we just want to determine an L2 reconstruction loss
        out = self.forward(torchInput)

        # l2_reconstruction_loss = nn.MSELoss()
        # loss = l2_reconstruction_loss(torchInput, out)

        l1_reconstruction_loss = nn.L1Loss()
        loss = l1_reconstruction_loss(torchInput, out)

        l2_lambda = 0.01
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        return loss