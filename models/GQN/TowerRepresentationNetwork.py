import torch
import torch.nn as nn

import torch.nn.functional as F

# Tower Representation Network

class TowerRepresentationNetwork(nn.Module):
    def __init__(self, fPoolingEnabled=False, *args, **kwargs):
        super(TowerRepresentationNetwork, self).__init__(*args, **kwargs)
        self.fPooling = fPoolingEnabled

        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256 + 7, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256 + 7, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        if(self.fPooling):
            self.pool = nn.AvgPool2d(16)


    def forward(self, input, in_view):
        out = input

        skip_input = F.relu(self.conv1(out))
        skip_out = F.relu(self.conv2(skip_input))

        representation = F.relu(self.conv3(skip_input))
        representation = F.relu(self.conv4(representation)) + skip_out

        # Broadcast
        in_view = in_view.view(in_view.size(0), 7, 1, 1).repeat(1, 1, 16, 16)

        # Residual and concatenate with broadcast view
        skip_input = torch.cat((representation, in_view), dim=1)
        skip_out = F.relu(self.conv5(skip_input))

        representation = F.relu(self.conv6(skip_input))
        representation = F.relu(self.conv7(representation)) + skip_out
        representation = F.relu(self.conv8(representation))

        # If pooling enabled pass the representation through the final pooling layer
        if (self.fPooling):
            representation = nn.AvgPool2d(representation)

        return representation