import torch
import torch.nn as nn

# ResnetBlock

class ResnetBlock(nn.Module):
    def __init__(self, n_filters=64, n_layers=4, kernel_size=7, fResidual=True, *args, **kwargs):
        super(ResnetBlock, self).__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.fResidual = fResidual

        self.net = []

        # Construct the network
        # TODO: Moar generalize
        for layer in range(self.n_layers):
            self.net.extend([
                nn.Conv2d(self.n_filters, self.n_filters // 2, 1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.n_filters // 2, self.n_filters // 2, self.kernel_size, stride=1, padding=self.inner_kernel // 2),
                nn.ReLU(),
                nn.Conv2d(self.n_filters // 2, self.n_filters, 1, stride=1, padding=0),
                nn.ReLU(),
            ])

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input
        for layer in self.net:
            out = layer(out)

        # More for debugging than anything, it is a resblock after all
        if(self.fResidual == True):
            return out + input
        else:
            return out

