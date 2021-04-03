import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

class BiGANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=1024, *args, **kwargs):
        super(BiGANGenerator, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.noise_distribution = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []

        layer_sizes = [self.input_dim] + [self.hidden_size] * self.n_layers

        i = 0
        for h1, h2 in zip(layer_sizes, layer_sizes[1:]):
            i += 1
            self.net.append(nn.Linear(h1, h2))
            if(i % 2 == 0):
                self.net.append(nn.BatchNorm(h2, affine=False))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(self.hidden_size, self.output_dim))
        self.net.append(nn.Tanh())

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)
        # out = out.reshape(-1, 1, 28, 28)
        return out

    def sample(self, n_samples):
        z = self.noise_distribution.sample([n_samples, self.input_dim])
        samples = self.forward(z)
        return samples, z


class BiGANDiscriminator(nn.Module):
    def __init__(self, input_dim, z_dim, output_dim, n_layers=2, hidden_size=1024, *args, **kwargs):
        super(BiGANGenerator, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []

        layer_sizes = [(self.input_dim + self.z_dim)] + [self.hidden_size] * self.n_layers

        i = 0
        for h1, h2 in zip(layer_sizes, layer_sizes[1:]):
            i += 1
            self.net.append(nn.Linear(h1, h2))
            if (i % 2 == 0):
                self.net.append(nn.BatchNorm(h2, affine=False))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Linear(self.hidden_size, self.output_dim))
        self.net.append(nn.Sigmoid())

        self.net = nn.ModuleList([*self.net])

    def forward(self, input, z):
        out = torch.cat((input, z), dim=1)
        # out = out.view(input.shape[0], -1)  # flatten
        for layer in self.net:
            out = layer(out)
        return out

class BiGANEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=1024, *args, **kwargs):
        super(BiGANEncoder, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []

        layer_sizes = [self.input_dim] + [self.hidden_size] * self.n_layers

        i = 0
        for h1, h2 in zip(layer_sizes, layer_sizes[1:]):
            i += 1
            self.net.append(nn.Linear(h1, h2))
            if (i % 2 == 0):
                self.net.append(nn.BatchNorm(h2, affine=False))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Linear(self.hidden_size, self.output_dim))

        self.net = nn.ModuleList([*self.net])

    def foward(self, input):
        out = input
        out = out.view(input.shape[0], -1)  # flatten
        for layer in self.net:
            out = layer(out)
        return out

class BiGAN(Model):
    def __init__(self, input_shape, output_dim, n_classes, latent_dim=50, n_layers=2, hidden_size=1024, *args, **kwargs):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        super(BiGAN, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        C, H, W = self.input_shape
        img_dim = C * H * W

        self.generator = BiGANGenerator(
            input_dim=self.latent_dim, output_dim=self.img_dim,
            n_layers=self.n_layers, hidden_size=self.hidden_size
        )

        self.critic = BiGANDiscriminator(
            input_dim=img_dim, z_dim=self.latent_dim, output_dim=1,
            n_layers=self.n_layers, hidden_size=self.hidden_size
        )

        self.encoder = BiGANEncoder(
            input_dim=img_dim, output_dim=self.latent_dim,
            n_layers=self.n_layers, hidden_size=self.hidden_size
        )

        # linear classifier
        self.linear_classifier = nn.Linear(self.output_dim, self.n_classes)
        #self.linear_optimizer = optim.Adam(self.linear_classifier.parameters(), lr=1e-3)

    def critic_loss(self, input):
        B, *_ = input.shape
        fake_data, z_fake_data = self.generator.sample(B)
        fake_data = fake_data.reshape(B, -1)
        z_real = self.encoder.forward(input).reshape(B, self.latent_dim)
        x_real = input.reshape(B, -1)
        c_loss = 0.5 * (self.critic.forward(x_real, z_real)).log().mean() - 0.5 * (1.0 - self.critic.forard(fake_data, z_fake_data)).log().mean()
        return c_loss



