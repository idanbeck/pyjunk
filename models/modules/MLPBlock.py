import torch
import torch.nn as nn

# MLPBlock

class MLPBlock(nn.Module):
    def __init__(self, dim_input, dim_output, dims_hidden, fResidual=False, *args,  **kwargs):
        super(MLPBlock, self).__init__(*args, **kwargs)

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dims_hidden = dims_hidden
        self.fResidual = fResidual

        if(self.fResidual == True and dim_input != dim_output):
            self.input_residual_transform = nn.Parameter(torch.Tensor(dim_input, dim_output), requires_grad=True)

        # Set up the network
        self.net = []
        self.ConstructNetwork()

    def ConstructNetwork(self):

        self.net = []

        layer_dims = [self.dim_input] + self.dims_hidden + [self.dim_output]
        i = 0
        for h_0, h_1 in zip(layer_dims[:], layer_dims[1:]):
            i += 1
            print("layer: %i [%i:%i]" % (i, h_0, h_1))
            self.net.extend([
                nn.Linear(h_0, h_1),
                nn.ReLU(),          # TODO: generalize activation fn.
            ])

        # Remove last ReLU
        self.net.pop()

        self.net = nn.ModuleList([*self.net])

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)

        if(self.fResidual == True):
            if(self.dim_input == self.dim_output):
                return out + input
            else:
                # This is hand wave-y, but apply a learned transform to input
                return out + torch.matmul(input, self.input_residual_transform)
        else:
            return out

