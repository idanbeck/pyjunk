import torch
import torch.nn as nn

# ResnetBlock

class ResnetBlock(nn.Module):
    def __init__(self, in_channels=64, n_filters=64, out_channels=64, n_layers=4, kernel_size=7, fResidual=True, *args, **kwargs):
        super(ResnetBlock, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.fResidual = fResidual

        # print("resnet block in:%d filter:%d out:%d" % (self.in_channels, self.n_filters, self.out_channels))

        self.net = []

        # Construct the network
        # TODO: Moar generalize
        for layer in range(self.n_layers):
            in_c = self.in_channels if layer == 0 else self.n_filters
            out_c = self.out_channels if layer >= (self.n_layers - 1) else self.n_filters

            self.net.extend([
                nn.Conv2d(in_c, self.n_filters, 1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.kernel_size // 2),
                nn.ReLU(),
                nn.Conv2d(self.n_filters, out_c, 1, stride=1, padding=0),
                nn.ReLU(),
            ])

        self.net.pop()
        self.net = nn.ModuleList([*self.net])

    def forward(self, input):

        # print("fwd: resnet block in:%d filter:%d out:%d" % (self.in_channels, self.n_filters, self.out_channels))

        out = input
        for layer in self.net:
            #print(out.shape)
            out = layer(out)

        # More for debugging than anything, it is a resblock after all
        if(self.fResidual == True):
            return out + input
        else:
            return out

