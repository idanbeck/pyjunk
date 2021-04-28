import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

import math

import repos.pyjunk.junktools.pytorch_utils  as ptu

# Convolutional VAE model

# Not really different than the normal ConvVAE encoder, but just for namespace
class SpatialConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        super(SpatialConvEncoder, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))
        self.hidden_spatial_extents = 8  # note: square only
        self.first_depth = 8  # 32
        self.ConstructModel()

    def ConstructModel(self):
        # input shape is h, w, c
        H, W, C = self.input_shape

        print(C)

        # # Construct the net
        # self.net = [
        #     nn.Conv2d(C, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(
        #         #4 * 4 * 256,
        #         8 * 8 * 256,
        #         2 * self.latent_dim
        #     ),
        # ]

        # Construct the net
        self.net = []
        current_spatial_dim = H
        current_depth_dim = C
        next_depth_dim = self.first_depth
        idx = 0

        self.net.extend([
            nn.Conv2d(C, next_depth_dim, 3, 1, 1),
            nn.ReLU(),
        ])
        current_depth_dim = next_depth_dim
        next_depth_dim = next_depth_dim * 2

        while (current_spatial_dim > self.hidden_spatial_extents):
            print("spatial %d -> %d depth %d -> %d" % (
                current_spatial_dim,
                current_spatial_dim // 2,
                current_depth_dim,
                next_depth_dim
            ))
            self.net.extend([
                nn.Conv2d(current_depth_dim, next_depth_dim, 3, 2, 1),
                nn.ReLU(),
            ])
            current_spatial_dim = current_spatial_dim // 2
            current_depth_dim = next_depth_dim
            next_depth_dim = next_depth_dim * 2

        # Final linear layer
        print("linear %d -> %d" % (
            current_spatial_dim * current_depth_dim,
            2 * self.latent_dim
        ))
        self.net.extend([
            nn.Flatten(),
            nn.Linear(
                # 4 * 4 * 256,
                (current_spatial_dim ** 2) * current_depth_dim,
                2 * self.latent_dim  # mean and std-dev so need two bruv
            ),
        ])

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)

        mu, log_std = out.chunk(2, dim=1)
        return mu, log_std

class SpatialConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape, kernel_size=5, *args, **kwargs):
        super(SpatialConvDecoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # output shape is h, w, c
        H, W, C = output_shape

        # instead of the transform, we simply broadcast
        # self.fc_layer = nn.Linear(
        #     self.latent_dim,
        #     #4 * 4 * 128
        #     1 * 1 * 128
        # )

        # construct the net
        self.net = [
            # nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2), # 64 x 64
            nn.Conv2d(self.latent_dim + 2, 128, kernel_size=self.kernel_size, stride=1, padding=self.padding),  # 64 x 64
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=self.kernel_size, stride=1, padding=self.padding),  # 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=1, padding=self.padding),  # 64 x 64
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        ]

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        B, *_ = input.shape
        out = input

        H, W, C = self.output_shape

        # first FC layer
        #out = self.fc_layer(out)

        # broadcast latent dimensions up to 64 x 64 and concatnate the positions
        out = input.view(-1, self.latent_dim, 1, 1).repeat(1, 1, H, W)
        width_positions = torch.linspace(-1, 1, W)
        height_positions = torch.linspace(-1, 1, H)
        x_pos, y_pos = torch.meshgrid(width_positions, height_positions)
        x_pos = x_pos.unsqueeze(0).repeat(B, 1, 1, 1).to(ptu.GetDevice())
        y_pos = y_pos.unsqueeze(0).repeat(B, 1, 1, 1).to(ptu.GetDevice())

        out = torch.cat((out, x_pos, y_pos), dim=1)

        # reshape to (4, 4, 128)
        #out = out.view(-1, 128, 1, 1).repeat(1, 1, 64, 64)
        #out = out.view(-1, 128, 8, 8)

        for layer in self.net:
            out = layer(out)
            #print(out.shape)

        return out

class SpatialConvVAE(Model):
    def __init__(self, input_shape, latent_dim, *args, **kwargs):

        # input shape is h, w, c
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))

        super(SpatialConvVAE, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        # Set up the encoder and decoder
        self.encoder = SpatialConvEncoder(
            self.input_shape,
            self.latent_dim).to(ptu.GetDevice())

        self.decoder = SpatialConvDecoder(
            self.latent_dim,
            self.input_shape,
            kernel_size=3).to(ptu.GetDevice())

    def loss(self, input):
        input = input.permute(0, 3, 1, 2)

        # shift into [-1, 1]
        out = input
        out = (out * 2.0) - 1.0

        # run the net
        mu_z, log_std_dev_z = self.encoder.forward(out)
        z = torch.randn_like(mu_z) * log_std_dev_z.exp() + mu_z
        z = z.to(ptu.GetDevice())
        x_tilda = self.decoder.forward(z)

        # print(x_tilda.shape)
        # print(out.shape)

        # Reconstruction loss is just MSE
        #reconstruction_loss = F.mse_loss(x_tilda, input, reduction='none').view(self.input_shape[0], -1).sum(1).mean()
        reconstruction_loss = F.mse_loss(x_tilda, out, reduction='none').view(self.input_shape[0], -1).sum(1).mean()

        # KL loss q(z|x) vs. N(0, I) (from VAE paper)
        kl_loss = (-0.5) * (1.0 + (2.0 * log_std_dev_z) - (mu_z ** 2) - (torch.exp(log_std_dev_z) ** 2))
        kl_loss = kl_loss.sum(1).mean()

        loss = kl_loss + reconstruction_loss

        return loss

    def sample(self, n_samples):
        images = []

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(ptu.GetDevice())
            samples = torch.clamp(self.decoder.forward(z), -1.0, 1.0)

        #samples = x.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5

        for x in samples:
            x = x.squeeze().permute(1, 2, 0) * 0.5 + 0.5
            newImage = image(torchBuffer=x)
            images.append(newImage)

        return images

    def forward_with_frame(self, frameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Run the model
        torchOutput = self.forward(torchImageBuffer)
        torchOutput = torchOutput.squeeze()

        # return an image
        return image(torchBuffer=torchOutput)


