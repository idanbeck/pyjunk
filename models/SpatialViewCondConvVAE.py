import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

import math

# Convolutional VAE model with a spatial decoder that also takes in the view to broadcast
# as well as conditional channels

# Not really different than the normal ConvVAE encoder, but just for namespace
class SpatialViewCondConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        super(SpatialViewCondConvEncoder, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))

        # input shape is h, w, c
        H, W, C = input_shape

        # TODO: programatic scale factor plz

        #scale_factor = 3
        #dim_mult = H // 3

        print(C)

        # Construct the net
        self.net = [
            nn.Conv2d(C, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 32 x 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                #4 * 4 * 256,
                8 * 8 * 256,
                2 * self.latent_dim
            ),
        ]

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)

        mu, log_std = out.chunk(2, dim=1)
        return mu, log_std

class SpatialViewCondConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape, kernel_size=3, view_dim=3, cond_dim=3, num_filters=128, num_layers=3, *args, **kwargs):
        super(SpatialViewCondConvDecoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.view_dim = view_dim
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.cond_dim = cond_dim

        # output shape is h, w, c
        H, W, C = output_shape

        # instead of the transform, we simply broadcast
        # self.fc_layer = nn.Linear(
        #     self.latent_dim,
        #     #4 * 4 * 128
        #     1 * 1 * 128
        # )

        # construct the net
        # TODO: Maybe do an instance norm in the future
        self.net = []

        first_layer_size = self.latent_dim + 2 + self.view_dim + self.cond_dim
        #print(first_layer_size)

        self.net.extend([
            nn.Conv2d(first_layer_size,
                      self.num_filters,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=self.padding),
            nn.ReLU(),
        ])

        for l in range(self.num_layers - 1):
            self.net.extend([
                #nn.BatchNorm2d(self.num_filters),
                nn.Conv2d(self.num_filters, self.num_filters,
                          kernel_size=self.kernel_size,
                          stride=1,
                          padding=self.padding),  # 64 x 64
                nn.ReLU(),
            ])

        self.net.extend([
            #nn.BatchNorm2d(self.num_filters),
            nn.Conv2d(self.num_filters, 3,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=self.padding),
            nn.Tanh()
        ])

        self.net = nn.ModuleList([*self.net])

    def forward(self, in_x, in_view, in_cond):
        B, *_ = in_x.shape


        # first FC layer
        #out = self.fc_layer(out)

        cond = in_cond

        # broadcast latent dimensions up to 64 x 64 and concatnate the positions
        out = in_x.view(-1, self.latent_dim, 1, 1).repeat(1, 1, 64, 64)
        width_positions = torch.linspace(-1, 1, 64)
        height_positions = torch.linspace(-1, 1, 64)
        x_pos, y_pos = torch.meshgrid(width_positions, height_positions)
        x_pos = x_pos.unsqueeze(0).repeat(B, 1, 1, 1)
        y_pos = y_pos.unsqueeze(0).repeat(B, 1, 1, 1)

        # broadcast the view
        broadcast_view = in_view.view(-1, self.view_dim, 1, 1).repeat(1, 1, 64, 64)

        out = torch.cat((out, cond, x_pos, y_pos, broadcast_view), dim=1)

        #print(out.shape)

        # reshape to (4, 4, 128)
        #out = out.view(-1, 128, 1, 1).repeat(1, 1, 64, 64)
        #out = out.view(-1, 128, 8, 8)

        for layer in self.net:
            out = layer(out)
            #print(out.shape)

        return out

class SpatialViewCondConvVAE(Model):
    def __init__(self, input_shape, latent_dim, view_dim=7, cond_dim=3, num_layers=12, num_filters=128, *args, **kwargs):
        super(SpatialViewCondConvVAE, self).__init__(*args, **kwargs)

        # input shape is h, w, c
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_dim_sqrt = int(math.sqrt(latent_dim))
        self.view_dim = view_dim
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.cond_dim = cond_dim

        # Set up the encoder and decoder
        self.encoder = SpatialViewCondConvEncoder(input_shape, self.latent_dim)
        self.decoder = SpatialViewCondConvDecoder(
            self.latent_dim,
            input_shape,
            kernel_size=3,
            view_dim=self.view_dim,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            cond_dim=cond_dim
        )

    def loss(self, in_x, in_view, in_cond):
        input = in_x.permute(0, 3, 1, 2)
        cond = in_cond.permute(0, 3, 1, 2)

        # shift into [-1, 1]
        out = input
        out = (out * 2.0) - 1.0

        # run the net
        mu_z, log_std_dev_z = self.encoder.forward(out)
        z = torch.randn_like(mu_z) * log_std_dev_z.exp() + mu_z
        x_tilda = self.decoder.forward(z, in_view, cond)

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
            z = torch.randn(n_samples, self.latent_dim)
            samples = torch.clamp(self.decoder.forward(z), -1.0, 1.0)

        #samples = x.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5

        for x in samples:
            x = x.squeeze().permute(1, 2, 0) * 0.5 + 0.5
            newImage = image(torchBuffer=x)
            images.append(newImage)

        return images

    def generate_with_cond_frame(self, cond_frame):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npCondFrameBuffer = cond_frame.GetNumpyBuffer()
        torchCondImageBuffer = torch.FloatTensor(npCondFrameBuffer)
        torchCondImageBuffer = torchCondImageBuffer.unsqueeze(0)
        torchCondImageBuffer = torchCondImageBuffer.permute(0, 3, 1, 2)

        with torch.no_grad():
            #z = torch.randn(1, self.latent_dim)
            z = torch.zeros(self.latent_dim).unsqueeze(0)
            torchViewBuffer = cond_frame.GetFrameCameraView().unsqueeze(0)
            sample = torch.clamp(
                self.decoder.forward(z, torchViewBuffer, torchCondImageBuffer),
                -1.0, 1.0
            )

        sample = sample.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        newImage = image(torchBuffer=sample)

        return newImage

    def generate_with_cond_frame_and_view(self, cond_frame, in_view):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npCondFrameBuffer = cond_frame.GetNumpyBuffer()
        torchCondImageBuffer = torch.FloatTensor(npCondFrameBuffer)
        torchCondImageBuffer = torchCondImageBuffer.unsqueeze(0)
        torchCondImageBuffer = torchCondImageBuffer.permute(0, 3, 1, 2)

        with torch.no_grad():
            #z = torch.randn(1, self.latent_dim)
            z = torch.zeros(self.latent_dim).unsqueeze(0)
            torchViewBuffer = in_view.GetViewPitchYawTorchTensor().unsqueeze(0)
            sample = torch.clamp(
                self.decoder.forward(z, torchViewBuffer, torchCondImageBuffer),
                -1.0, 1.0
            )

        sample = sample.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        newImage = image(torchBuffer=sample)

        return newImage

    def forward_with_frame(self, frameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Retrieve frame view
        torchQueryViewBuffer = frameObject.GetFrameCameraView().unsqueeze(0)

        # Run the model
        torchOutput = self.forward(torchImageBuffer, torchQueryViewBuffer)
        torchOutput = torchOutput.squeeze()

        # return an image
        return image(torchBuffer=torchOutput)

    def loss_with_frame(self, frameObject, condFrameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npCondFrameBuffer = condFrameObject.GetNumpyBuffer()
        torchCondImageBuffer = torch.FloatTensor(npCondFrameBuffer)
        torchCondImageBuffer = torchCondImageBuffer.unsqueeze(0)

        # Retrieve frame view
        torchQueryViewBuffer = frameObject.GetFrameCameraView().unsqueeze(0)

        # Run the model
        torchLoss = self.loss(
            in_x=torchImageBuffer,
            in_view=torchQueryViewBuffer,
            in_cond=torchCondImageBuffer
        )

        # return an image
        return torchLoss

