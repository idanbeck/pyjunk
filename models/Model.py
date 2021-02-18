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
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Run the model
        torchOutput = self.forward(torchImageBuffer)
        torchOutput = torchOutput.squeeze()

        # return an image
        return image(torchBuffer=torchOutput)

    def forward_with_numpy_buffer(self, npBuffer):
        torchBuffer = torch.FloatTensor(npBuffer)

        # Run the model
        torchOutput = self.forward(torchBuffer)

        images = []

        for i, torchOutput in enumerate(torchOutput):
            newImage = image(torchBuffer=torchOutput)
            images.append(newImage)

        # Return array of images
        return images

    def forward_with_frameset(self, framesetObject):
        # Grab the numpy buffer from the frameset (BS, H, W, C)
        npFramesetBuffer = framesetObject.GetNumpyBuffer()

        return self.forward_with_numpy_buffer(npFramesetBuffer)

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
        torchSourceImageBuffer = torch.FloatTensor(npSourceFrameBuffer).unsqueeze(0)

        npTargetFrameBuffer = targetFrame.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer).unsqueeze(0)

        # loss with target
        torchLoss = self.loss_with_target(torchSourceImageBuffer, torchTargetImageBuffer)

        return torchLoss

    def loss_with_frameset_and_target(self, sourceFrameset, targetFrameset):
        # Grab the numpy buffer from the source frameset (BS, H, W, C)
        npSourceFramesetBuffer = sourceFrameset.GetNumpyBuffer()
        torchSourceImageBuffer = torch.FloatTensor(npSourceFramesetBuffer)

        # Grab the numpy buffer from the target frameset (BS, H, W, C)
        npTargetFramesetBuffer = targetFrameset.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFramesetBuffer)

        # loss with target
        torchLoss = self.loss_with_target(torchSourceImageBuffer, torchTargetImageBuffer)

        return torchLoss