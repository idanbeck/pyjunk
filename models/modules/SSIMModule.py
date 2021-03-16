import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def CreateGaussianWindow(window_size, channels, sigma):
    # Set up gaussian kernel
    gaussian_kernel = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
    ])

    # normalize
    gaussian_kernel /= gaussian_kernel.sum()

    # get to rank 2
    gaussian_kernel = gaussian_kernel.unsqueeze(1)

    # Auto-correlate
    gaussian_kernel = gaussian_kernel.mm(gaussian_kernel.t()).float()

    # get to [1, 1, size, size]
    gaussian_kernel = gaussian_kernel

    gaussian_kernel = torch.autograd.Variable(
        gaussian_kernel.expand(channels, 1, window_size, window_size).contiguous()
    )

    return gaussian_kernel

class SSIMModule(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, c1=1e-4, c2=3e-4, fSizeAverage=True, *args, **kwargs):
        super(SSIMModule, self).__init__(*args, **kwargs)
        self.window_size = window_size
        self.sigma = sigma
        self.channels = 3
        self.c1 = c1
        self.c2 = c2
        self.fSizeAverage = fSizeAverage

        self.window = CreateGaussianWindow(
            self.window_size,
            self.channels,
            self.sigma
        )

    def forward(self, x, y):
        B, C, H, W = x.shape

        c1 = self.c1
        c2 = self.c2

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=self.channels)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=self.channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sig_x_sq = F.conv2d(x*x, self.window, padding=self.window_size//2, groups=self.channels)
        sig_y_sq = F.conv2d(y*y, self.window, padding=self.window_size//2, groups=self.channels)
        sig_xy = F.conv2d(x*y,  self.window, padding=self.window_size//2, groups=self.channels)

        ssim =  ((2.0 * mu_xy + c1)*(2.0*sig_xy + c2))
        ssim /= (mu_x_sq + mu_y_sq + c1) * (sig_x_sq + sig_y_sq + c2)

        if(self.fSizeAverage == True):
            return ssim.mean()
        else:
            return ssim.mean(1).mean(1).mean(1)


