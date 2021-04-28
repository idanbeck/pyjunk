import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
import torch.optim as optim
import repos.pyjunk.junktools.pytorch_utils  as ptu

# CycleGAN Model

class CycleGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_shape, n_layers=2, n_filters=64, *args, **kwargs):
        super(CycleGANGenerator, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.stride = 1
        self.noise_distribution = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []
        H, W, C = self.output_shape

        self.net.extend([
            nn.ConvTranspose2d(self.latent_dim, self.n_filters * 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.n_filters * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_filters * 4, self.n_filters * 2, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.n_filters * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_filters * 2, self.n_filters, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.n_filters),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_filters, C, kernel_size=4, stride=2),
            nn.Tanh(),
        ])

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)
        # out = out.reshape(-1, 1, 28, 28)
        return out

    def sample(self, n_samples):
        #z = self.noise_distribution.sample([n_samples, self.input_dim])
        z = (torch.rand(n_samples, self.latent_dim).to(ptu.GetDevice()) - 0.5) * 2.0
        z = z.view(n_samples, self.latent_dim, 1, 1)
        samples = self.forward(z)
        return samples, z


class CycleGANDiscriminator(nn.Module):
    def __init__(self, input_shape, n_layers=2, n_filters=1024, *args, **kwargs):
        super(CycleGANDiscriminator, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.n_layers = n_layers
        self.n_filters = n_filters

        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = []

        H, W, C = self.input_shape

        self.net.extend([
            # module 1
            nn.Conv2d(C, self.n_filters, kernel_size=4, stride=2),
            nn.BatchNorm2d(self.n_filters),
            nn.LeakyReLU(0.2),

            # module 2
            nn.Conv2d(self.n_filters, self.n_filters * 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(self.n_filters * 2),
            nn.LeakyReLU(0.2),

            # final layer
            nn.Conv2d(self.n_filters * 2, 1, kernel_size=4, stride=2)
        ])
        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        B, *_ = input.shape
        #out = torch.cat((z, input), dim=1)
        # out = out.view(input.shape[0], -1)  # flatten

        out = input.squeeze().unsqueeze(dim=1)

        for layer in self.net:
            out = layer(out)

        # flatten output
        out = out.view(B, -1)
        return out

class CycleGAN(Model):
    def __init__(self, input_shape, latent_dim=64, n_layers=2, n_filters=1024, *args, **kwargs):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_filters = n_filters
        super(CycleGAN, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        C, H, W = self.input_shape
        img_dim = C * H * W

        self.generator = CycleGANGenerator(
            output_shape = self.input_shape,
            latent_dim = self.latent_dim,
            n_layers=self.n_layers,
            n_filters = self.n_filters
        ).to(ptu.GetDevice())

        self.discriminator = CycleGANDiscriminator(
            input_shape= self.input_shape,
            n_layers=self.n_layers,
            n_filters=self.n_filters
        ).to(ptu.GetDevice())

        # Initialize the weights to the normal distribution with mean 0 and standard deviation 0.02
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

        self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

    def discriminator_loss(self, input):
        B, *_ = input.shape
        criterion = nn.BCEWithLogitsLoss()
        fake_data, z_fake_data = self.generator.sample(B)
        x_real = input

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

        #g_loss = self.discriminator.forward(fake_data).log().mean()
        disc_fake_pred = self.discriminator.forward(fake_data)
        g_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)).mean()
        #print(g_loss)

        return g_loss

    def SetupGANOptimizers(self, solver):
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=solver.lr, betas=solver.betas, eps=solver.eps
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
