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

        in_H, in_W, in_C = self.input_shape
        out_H, out_W, out_C = self.output_shape

        print("in:%d out:%d channels" % (in_C, out_C))

        # # First block
        # self.net.extend([
        #     nn.BatchNorm2d(C),
        #     ResnetBlock(
        #         in_channels=C,
        #         n_filters=self.n_filters,
        #         out_channels=self.n_filters,
        #         n_layers=self.n_layers,
        #         kernel_size=self.kernel_size,
        #         fResidual=self.fResidual),
        #     nn.ReLU()  # TODO: moar generalize
        # ])

        for block in range(self.n_blocks):
            #in_c = in_C if block == 0 else self.n_filters
            in_c = in_C
            out_c = out_C if block >= (self.n_blocks - 1) else in_C

            # mismatch between target output channels and input
            # TODO: create a learned mapping (see MLP blocks for this)
            fResidual = False if block >= (self.n_blocks - 1) else self.fResidual

            self.net.extend([
                #nn.BatchNorm2d(self.n_filters),
                nn.BatchNorm2d(in_c),
                ResnetBlock(
                    #in_channels=self.n_filters,
                    in_channels=in_c,
                    n_filters=self.n_filters,
                    out_channels=out_c,
                    #out_channels=C,
                    n_layers=self.n_layers,
                    kernel_size=self.kernel_size,
                    fResidual=fResidual),
                nn.ReLU()  # TODO: moar generalize
            ])

        # # Last block
        # self.net.extend([
        #     nn.BatchNorm2d(C),
        #     ResnetBlock(
        #         in_channels=self.n_filters,
        #         n_filters=self.n_filters,
        #         out_channels=C,
        #         n_layers=self.n_layers,
        #         kernel_size=self.kernel_size,
        #         fResidual=self.fResidual),
        #     nn.ReLU()  # TODO: moar generalize
        # ])

        # eliminate the last baddie
        self.net.pop()
        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input.permute(0, 3, 1, 2)

        #print(out.shape)
        for block in self.net:
            #print(out.shape)
            out = block(out)

        out = out.permute(0, 2, 3, 1)
        return out

    def loss(self, torchInput):
        # Ultimately we just want to determine an L2 reconstruction loss
        out = self.forward(torchInput)

        # l2_reconstruction_loss = nn.MSELoss()
        # loss = l2_reconstruction_loss(torchInput, out)

        l1_reconstruction_loss = nn.L1Loss()
        loss = l1_reconstruction_loss(torchInput, out)

        l2_lambda = 0.005
        l2_reg = torch.tensor(0.)
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

        l2_lambda = 0.005
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        return loss