import torch
import torch.nn
import torch.nn.functional as F

from repos.pyjunk.models.modules.ConvLSTM import Conv2dLSTMCell

# Generative Query Network Cores

class InferenceCore(nn.Module):
    def __init__(self, id=0, *args, **kwargs):
        super(InferenceCore, self).__init__(*args, **kwargs)
        self.id = id

        self.downsample_input = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_view = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_reconstruction = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3 + 7 + 256 + (2 * 128), kernel_size=5, stride=1, padding=2)

    def forward(self,
                input, in_view, representation,
                cell_state_inf, hidden_state_inf,
                hidden_state_gen, u):

        input = self.downsample_input(input)
        in_view = self.upsample_view(in_view.view(-1, 7, 1, 1))
        if representation.size(2) != hidden_state_inf.size(2):
            representation = self.upsample_reconstruction(representation)
        u = self.downsample_u(u)

        # Run through conv LSTM
        lstm_input = torch.cat((input, in_view, representation, hidden_state_gen, u), dim=1)
        cell_state_inf, hidden_state_inf = self.core(lstm_input, (cell_state_inf, hidden_state_inf))

        return (cell_state_inf, hidden_state_inf)


class GenerationCore(nn.Module):
    def _init__(self, id=0, *args, **kwargs):
        super(GenerationCore, self).__init__(*args, **kwargs)
        self.id = id

        self.upsample_view = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_reconstruction = nn.ConvTranspose2d2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3 + 7 + 256 + (2 * 128), kernel_size=5, stride=1, padding=2)
        self.upsample_hidden_state = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)

    def forward(self,
                in_view, representation,
                cell_state, hidden_state,
                u, z):
        in_view = self.upsample_view(in_view.view(-1, 7, 1, 1))
        if representation.size(2) != hidden_state.size(2):
            representation = self.upsample_reconstruction(representation)

        lstm_input = torch.cat((in_view, representation, z), dim=1)
        cell_state, hidden_state = self.core(lstm_input, (cell_state, hidden_state))
        u = self.upsample_hidden_state(hidden_state) + u

        return (cell_state, hidden_state)