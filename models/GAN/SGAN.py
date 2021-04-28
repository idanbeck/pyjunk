import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
import torch.optim as optim
import repos.pyjunk.junktools.pytorch_utils  as ptu

# Standard GAN baseline

class SGANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=1024, *args, **kwargs):
        super(SGANGenerator, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.noise_distribution = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size * 2),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.ReLU(),

            nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
            nn.BatchNorm1d(self.hidden_size * 4),
            nn.ReLU(),

            nn.Linear(self.hidden_size * 4, self.hidden_size * 8),
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.ReLU(),

            nn.Linear(self.hidden_size * 8, self.output_dim),
            nn.Tanh()
        )
        # layer_sizes = [self.input_dim] + [self.hidden_size] * self.n_layers
        #
        # i = 0
        # for h1, h2 in zip(layer_sizes, layer_sizes[1:]):
        #     i += 1
        #     self.net.append(nn.Linear(h1, h2))
        #     if(i % 2 == 0):
        #         self.net.append(nn.BatchNorm1d(h2, affine=False))
        #     self.net.append(nn.ReLU())
        #
        # self.net.append(nn.Linear(self.hidden_size, self.output_dim))
        # self.net.append(nn.Tanh())
        #
        # self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)
        # out = out.reshape(-1, 1, 28, 28)
        return out

    def sample(self, n_samples):
        #z = self.noise_distribution.sample([n_samples, self.input_dim])
        z = (torch.rand(n_samples, self.input_dim).to(ptu.GetDevice()) - 0.5) * 2.0
        samples = self.forward(z)
        return samples, z


class SGANDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_size=1024, *args, **kwargs):
        super(SGANDiscriminator, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            #nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size * 8, 1),
            nn.Sigmoid()
        )
        # layer_sizes = [(self.input_dim + self.z_dim)] + [self.hidden_size] * self.n_layers
        #
        # i = 0
        # for h1, h2 in zip(layer_sizes, layer_sizes[1:]):
        #     i += 1
        #     self.net.append(nn.Linear(h1, h2))
        #     if (i % 2 == 0):
        #         self.net.append(nn.BatchNorm1d(h2, affine=False))
        #     self.net.append(nn.LeakyReLU(0.2))
        #
        # self.net.append(nn.Linear(self.hidden_size, self.output_dim))
        # self.net.append(nn.Sigmoid())
        #
        # self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        #out = torch.cat((z, input), dim=1)
        # out = out.view(input.shape[0], -1)  # flatten
        out = input
        for layer in self.net:
            out = layer(out)
        return out

class SGAN(Model):
    def __init__(self, input_shape, n_classes, latent_dim=50, n_layers=2, hidden_size=1024, *args, **kwargs):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        super(SGAN, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        C, H, W = self.input_shape
        img_dim = C * H * W

        self.generator = SGANGenerator(
            input_dim=self.latent_dim,
            output_dim=img_dim,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size
        ).to(ptu.GetDevice())

        self.discriminator = SGANDiscriminator(
            input_dim=img_dim,
            output_dim=1,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size
        ).to(ptu.GetDevice())

    def discriminator_loss(self, input):
        B, *_ = input.shape
        criterion = nn.BCEWithLogitsLoss()
        fake_data, z_fake_data = self.generator.sample(B)
        fake_data = fake_data.reshape(B, -1)
        x_real = input.reshape(B, -1)

        disc_fake_pred = self.discriminator.forward(fake_data.detach())
        disc_real_pred = self.discriminator.forward(x_real)

        # d_loss = 0.5 * (self.discriminator.forward(x_real)).log().mean()
        # d_loss -= 0.5 * (1.0 - self.discriminator.forward(fake_data)).log().mean()
        d_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        d_loss += criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        d_loss /= 2.0
        d_loss = d_loss.mean()

        #print(d_loss)
        return d_loss

    def generator_loss(self, input):
        B, *_ = input.shape
        criterion = nn.BCEWithLogitsLoss()
        fake_data, z_fake_data = self.generator.sample(B)
        fake_data = fake_data.reshape(B, -1) # flatten
        #g_loss = self.discriminator.forward(fake_data).log().mean()
        disc_fake_pred = self.discriminator.forward(fake_data)
        g_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)).mean()
        #print(g_loss)

        return g_loss

    def SetupGANOptimizers(self, solver):
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=solver.lr, betas=solver.betas, eps=solver.eps, weight_decay=2.5e-5
        )

        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            #lr=2e-4, betas=(0, 0.9), eps=solver.eps, weight_decay=2.5e-5
            lr=solver.lr, betas=solver.betas, eps=solver.eps
        )

    def SetupGANSchedulers(self, solver):
        self.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer,
            lambda epoch: (solver.epochs - epoch) / solver.epochs,
            last_epoch=-1
        )

        self.discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_optimizer,
            lambda epoch: (solver.epochs - epoch) / solver.epochs,
            last_epoch=-1
        )

    def sample(self, n_samples=1000):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            samples, z = self.generator.sample(n_samples)
            samples = (samples.reshape(-1, 28, 28, 1)) * 0.5 + 0.5

        return samples
