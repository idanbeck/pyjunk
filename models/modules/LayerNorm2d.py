import torch
import torch.nn as nn

# Layer Normalization 2d

class LayerNorm2d(nn.Module):
    def __init__(self, n_channels, eps=1e-5, affine=True, *args, **kwargs):
        super(LayerNorm2d, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine

        self.ConstructModule()

    def ConstructModule(self):
        if(self.affine):
            self.gamma = nn.Parameter(torch.rand(self.n_channels))
            self.beta = nn.Parameter(torch.zeros(self.n_channels))

    def forward(self, x_input, w_style):
        mean = x_input.flatten(1).mean(1).reshape(-1, 1, 1, 1)
        std_dev = x_input.flatten(1).std(1).reshape(-1, 1, 1, 1)

        out = (x_input - mean) / (std_dev + self.eps)

        if(self.affine):
            out = out * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1)

        return out