import torch
import torch.nn as nn

# Base Model class

from repos.pyjunk.junktools.image import image

class Model(nn.Module):
    def __init__(self, *args,  **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.net = []

    def VisualizeModel(self):
        pass

    # This stub should be overridden
    def forward(self, torchInput):
        pass

    def forward_with_frame(self, frameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)

        # Run the model
        torchOutput = self.forward(torchImageBuffer)

        # return an image
        return image(torchBuffer=torchOutput)

    def forward_with_image(self, imageObject):
        # Convert to torch tensor
        npImageBuffer = imageObject.npImageBuffer
        torchImageBuffer = torch.FloatTensor(npImageBuffer)

        # Run the model
        torchOutput = self.forward(torchImageBuffer)

        # return an image
        return image(torchBuffer=torchOutput)

    def loss_with_image(self, imageObject):
        # Convert to torch tensor
        npImageBuffer = imageObject.npImageBuffer
        torchImageBuffer = torch.FloatTensor(npImageBuffer)

        # loss
        torchLoss = self.loss(torchImageBuffer)

        return torchLoss

    def loss_with_frame_and_target(self, sourceFrame, targetFrame):
        npSourceFrameBuffer = sourceFrame.GetNumpyBuffer()
        torchSourceImageBuffer = torch.FloatTensor(npSourceFrameBuffer)

        npTargetFrameBuffer = targetFrame.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer)

        # loss with target
        torchLoss = self.loss_with_target(torchSourceImageBuffer, torchTargetImageBuffer)

        return torchLoss