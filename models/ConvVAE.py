import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model

# Convolutional VAE model

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        super(ConvEncoder, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        C, H, W = input_shape

        # Construct the net
        self.net = [
            nn.Conv2d(C, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 2 * self.latent_dim),
        ]

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)

        mu, log_std = out.chunk(2, dim=1)
        return mu, log_std

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape, *args, **kwargs):
        super(ConvDecoder, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        C, H, W = output_shape

        self.fc_layer = nn.Linear(self.latent_dim, 4 * 4 * 128)

        # construct the net
        self.net = [
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 4, 2, 1),  # 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
        ]

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input

        # first FC layer
        out = self.fc_layer(out)

        # reshape to (4, 4, 128)
        out = out.view(-1, 128, 4, 4)

        for layer in self.net:
            out = layer(out)

        return out

class ConvVAE(Model):
    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        super(ConvVAE, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Set up the encoder and decoder
        self.encoder = ConvEncoder(input_shape, self.latent_dim)
        self.decoder = ConvDecoder(self.latent_dim, input_shape)

    def loss(self, input):
        # shift into [-1, 1]
        out = input
        out = (out * 2.0) - 1.0

        # run the net
        mu_z, log_std_dev_z = self.encoder.forward(out)
        z = torch.randn_like(mu_z) * log_std_dev_z.exp() + mu_z
        x_tilda = self.decoder.forward(z)

        # Reconstruction loss is just MSE
        reconstruction_loss = F.mse_loss(x_tilda, input, reduction='none').view(self.input_shape[0], -1).sum(1).mean()

        # KL loss q(z|x) vs. N(0, I) (from VAE paper)
        kl_loss = (-0.5) * (1.0 + (2.0 * log_std_dev_z) - (mu_z ** 2) - (torch.exp(log_std_dev_z) ** 2))
        kl_loss = kl_loss.sum(1).mean()

        loss = kl_loss + reconstruction_loss

        return loss

    #def sample(self, n_samples):
