import torch
import torch.nn as nn

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
import torch.optim as optim
import repos.pyjunk.junktools.pytorch_utils  as ptu

# WGAN_GP baseline

# note - using the SNGAN CIFAR-10 arch
# TODO: Move/generalize the modules

class DepthToSpace(nn.Module):
    def __init__(self, block_size, *args, **kwargs):
        super(DepthToSpace, self).__init__(*args, **kwargs)
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        out = input.permute(0, 2, 3, 1)
        B, H, W, D = out.shape
        s_depth = int(D / self.block_size_sq)
        s_width = int(W * self.block_size)
        s_height = int(H * self.block_size)
        t_1 = out.reshape(B, H, W, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(B, H, s_width, s_depth) for t_t in spl]
        out = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(B, s_height, s_width, s_depth)
        out = out.permute(0, 3, 1, 2)
        return out

class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, *args, **kwargs):
        super(Upsample_Conv2d, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.depthToSpace = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

    def forward(self, input):
        out = input
        out = torch.cat([out, out, out, out], dim=1)
        out = self.depthToSpace.forward(out)
        out = self.conv.forward(out)
        return out

class SpaceToDepth(nn.Module):
    def __init__(self, block_size, *args, **kwargs):
        super(SpaceToDepth, self).__init__(*args, **kwargs)
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        out = input.permute(0, 2, 3, 1)
        (B, H, W, D) = out.shape
        d_depth = D * self.block_size_sq
        d_width = int(W / self.block_size)
        d_height = int(H / self.block_size)
        t_1 = out.split(self.block_size, 2)
        stack = [t_t.reshape(B, d_height, d_depth) for t_t in t_1]
        out = torch.stack(stack, 1)
        out = out.permute(0, 2, 1, 3)
        out = out.permute(0, 3, 1, 2)
        return out

class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True, *args, **kwargs):
        super(Downsample_Conv2d, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.spaceToDepth = SpaceToDepth(block_size=2)
        self.conv = nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride = self.stride,
            padding = self.padding,
            bias = self.bias
        )

    def forward(self, input):
        out = input
        out = self.spaceToDepth.forward(out)
        out = sum(out.chunk(4, dim=1)) / 4.0
        #out = torch.sum(out.chunk(4, dim=1)) / 4.0
        out = self.conv(out)
        return out

class ResBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256, *args, **kwargs):
        super(ResBlockUp, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.net = [
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(),
            nn.Conv2d(
                self.in_dim,
                self.n_filters,
                kernel_size=self.kernel_size,
                stride=1, padding=1
            )
        ]
        self.net = nn.ModuleList([*self.net])

        self.upsample_residual = Upsample_Conv2d(
            self.n_filters, self.n_filters, self.kernel_size, padding=1
        )
        self.upsample_shortcut = Upsample_Conv2d(
            self.in_dim, self.n_filters, kernel_size=(1,1), padding=0
        )

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)

        residual = self.upsample_residual(out)
        shortcut = self.upsample_shortcut(input)
        return residual + shortcut

class ResBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256, *args, **kwargs):
        super(ResBlockDown, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.net = [
            nn.ReLU(),
            nn.Conv2d(self.in_dim, self.n_filters, self.kernel_size, padding=1),
            nn.ReLU()
        ]
        self.net = nn.ModuleList([*self.net])

        self.downsample_residual = Downsample_Conv2d(
            self.n_filters, self.n_filters, kernel_size=self.kernel_size, padding = 1
        )

        self.downsample_shortcut = Downsample_Conv2d(
            self.in_dim, self.n_filters, kernel_size=(1, 1), padding = 0
        )

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)

        residual = self.downsample_residual(out)
        shortcut = self.downsample_shortcut(input)
        return residual + shortcut


class WGAN_GPGenerator(nn.Module):
    def __init__(self, n_filters, *args, **kwargs):
        super(WGAN_GPGenerator, self).__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.noise = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.ConstructNetwork()

    def ConstructNetwork(self):

        self.fc = nn.Linear(self.n_filters, 4 * 4 * 256)
        self.net = [
            ResBlockUp(in_dim=256, n_filters=self.n_filters),
            ResBlockUp(in_dim=self.n_filters, n_filters=self.n_filters),
            ResBlockUp(in_dim=self.n_filters, n_filters=self.n_filters),
            nn.BatchNorm2d(self.n_filters),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ]
        self.net = nn.ModuleList([*self.net])


    def forward(self, input):
        out = input
        out = self.fc(out).reshape(-1, 256, 4, 4)
        for layer in self.net:
            out = layer(out)
        return out

    def sample(self, n_samples):
        z = self.noise.sample([n_samples, self.n_filters]).to(ptu.GetDevice())
        samples = self.forward(z)
        return samples, z


class WGAN_GPDiscriminator(nn.Module):
    def __init__(self, n_filters=128, *args, **kwargs):
        super(WGAN_GPDiscriminator, self).__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.out_dim = 1
        self.ConstructNetwork()

    def ConstructNetwork(self):
        self.net = [
            ResBlockDown(in_dim=3, n_filters=self.n_filters),
            ResBlockDown(in_dim=self.n_filters, n_filters=self.n_filters),
            ResBlockDown(in_dim=self.n_filters, n_filters=self.n_filters),
            nn.ReLU()
        ]
        self.net = nn.ModuleList([*self.net])
        self.fc = nn.Linear(self.n_filters, self.out_dim)


    def forward(self, input):
        #out = torch.cat((z, input), dim=1)
        # out = out.view(input.shape[0], -1)  # flatten
        out = input
        for layer in self.net:
            #print(out.shape)
            out = layer(out)
        out = torch.sum(out, dim=(2, 3))
        out = self.fc(out)
        return out

class WGAN_GP(Model):
    def __init__(self, input_shape, n_filters=128, *args, **kwargs):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.lambda_gp = 10.0

        super(WGAN_GP, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        C, H, W = self.input_shape
        img_dim = C * H * W

        self.generator = WGAN_GPGenerator(
            n_filters=self.n_filters
        ).to(ptu.GetDevice())

        self.discriminator = WGAN_GPDiscriminator(
            n_filters=self.n_filters
        ).to(ptu.GetDevice())

    def GradientPenalty(self, real_data, fake_data):
        # B, *_ = real_data.shape
        #
        # # interpolation
        # eps = torch.rand(B, 1, 1, 1).to(ptu.GetDevice())
        # eps = eps.expand_as(real_data)
        # interpolated_data = eps * real_data + (1.0 - eps) * fake_data
        # interpolated_data.requires_grad = True
        #
        # d_output = self.discriminator(interpolated_data)
        # gradients = torch.autograd.grad(outputs=d_output, inputs=interpolated_data,
        #                                 grad_outputs=torch.ones(d_output.size()).to(ptu.GetDevice()),
        #                                 create_graph=True, retain_graph=True)[0]
        #
        # gradients = gradients.reshape(B, -1)
        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # return ((gradients_norm - 1.0) ** 2.0).mean()
        batch_size = real_data.shape[0]

        # print(real_data.shape)
        # print(fake_data.shape)

        # Calculate interpolation
        eps = torch.rand(batch_size, 1, 1, 1).to(ptu.GetDevice())
        eps = eps.expand_as(real_data)
        interpolated = eps * real_data.data + (1 - eps) * fake_data.data
        interpolated.requires_grad = True

        d_output = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=d_output, inputs=interpolated,
                                        grad_outputs=torch.ones(d_output.size()).to(ptu.GetDevice()),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.reshape(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def discriminator_loss(self, input):
        input = input.permute(0, 3, 1, 2)
        B, *_ = input.shape
        #criterion = nn.BCEWithLogitsLoss()

        fake_data, z_fake_data = self.generator.sample(B)
        x_real = input

        disc_fake_pred = self.discriminator.forward(fake_data.detach())
        disc_real_pred = self.discriminator.forward(x_real)
        gp = self.GradientPenalty(x_real, fake_data)

        d_loss = disc_fake_pred.mean() - disc_real_pred.mean() + (self.lambda_gp * gp)

        # d_loss = 0.5 * (self.discriminator.forward(x_real)).log().mean()
        # d_loss -= 0.5 * (1.0 - self.discriminator.forward(fake_data)).log().mean()
        # d_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        # d_loss += criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        #d_loss /= 2.0
        #d_loss = d_loss.mean()

        #print(d_loss)
        return d_loss

    def generator_loss(self, input):
        B, *_ = input.shape
        criterion = nn.BCEWithLogitsLoss()
        fake_data, z_fake_data = self.generator.sample(B)

        disc_fake_pred = self.discriminator.forward(fake_data)
        g_loss = -disc_fake_pred.mean()

        #g_loss = self.discriminator.forward(fake_data).log().mean()
        #g_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)).mean()
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
            samples = (samples.permute(0, 2, 3, 1)) * 0.5 + 0.5
            #samples = ptu.get_numpy(samples) * 0.5 + 0.5

        return samples
