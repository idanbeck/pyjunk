import torch
import torch.nn as nn

# Adaptive Instance Normalization 2d

class AdaptiveInstanceNorm2d(nn.Module):
    @staticmethod
    def mlp(self, in_dim, h_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim)
        )

    def __init__(self, n_channels, style_dim=8, hidden_dim=256, *args, **kwargs):
        super(AdaptiveInstanceNorm2d, self).__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim

        self.ConstructModule()

    def ConstructModule(self):
        self.instance_norm = nn.InstanceNorm(self.n_channels, affine=False)
        self.style_scale_transform = self.mlp(self.style_dim, self.hidden_dim, self.n_channels)
        self.style_shift_transform = self.mlp(self.style_dim, self.hidden_dim, self.n_channels)

    def forward(self, x_input, w_style):
        '''
        Output of AdaIN first normalizes the input image, transforms the style into a respective
        scale and shift, then applies this to the normalized input image
        '''
        normalized = self.instance_norm(x_input)
        style_scale = self.style_scale_transform(w_style)[:, :, None, None]
        style_shift = self.style_shift_transform(w_style)[:, :, None, None]
        transformed = style_scale * normalized + style_shift
        return transformed