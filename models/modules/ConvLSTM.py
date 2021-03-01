import torch
import torch.nn as nn

# Conv LSTM Module

class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, *args, **kwargs):
        super(Conv2dLSTMCell, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        convParams = dict(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.forget = nn.Conv2d(self.in_channels + self.out_channels, self.out_channels, **convParams)
        self.input = nn.Conv2d(self.in_channels + self.out_channels, self.out_channels, **convParams)
        self.output = nn.Conv2d(self.in_channels + self.out_channels, self.out_channels, **convParams)
        self.state = nn.Conv2d(self.in_channels + self.out_channels, self.out_channels, **convParams)

    def forward(self, input, states):
        cell_state, hidden_state = states

        input = torch.cat((hidden_state, input), dim=1)

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate = torch.tanh(self.state(input))

        # Internal state
        cell_state = forget_gate * cell_state + input_gate * state_gate
        hidden_state = output_gate * torch.tanh(cell_state)

        return (cell_state, hidden_state)
