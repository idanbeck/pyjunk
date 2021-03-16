import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

from repos.pyjunk.models.ConvUNet import ConvUNet

import math

class SSFeatExtract(nn.Module):
    def __init__(self, input_shape, num_filters=32, out_features=8, *args, **kwargs):
        super(SSFeatExtract, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.out_features = out_features

        H, W, C = input_shape

        self.net = []

        self.net.extend([
            nn.Conv2d(C, self.num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_filters, self.out_features, 3, 1, 1),
            nn.LeakyReLU(),
        ])

        self.net = nn.ModuleList(*[self.net])

    def forward(self, input):
        out = input

        for layer in self.net:
            out = layer(out)

        # concat input features with learned extracted features
        out = torch.cat((input, out), dim=1)

        return out

class SupersamplingUNet(Model):
    def __init__(self, input_shape, output_shape, scale=3, num_filters=32, *args, **kwargs):
        super(SupersamplingUNet, self).__init__(*args, **kwargs)

        # input shape is h, w, c
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters
        self.out_features = 8

        in_H, in_W, in_C = input_shape
        out_H, out_W, out_C = output_shape

        self.scale_factor = out_H // in_H

        # Feature Extraction Network
        self.feat_extract = SSFeatExtract(
            input_shape=input_shape,
            num_filters=32,
            out_features=self.out_features
        )

        # upsampling (TODO: Zero upsampling implementation)
        self.upsampling = nn.Upsample(
            #size=self.output_shape,
            scale_factor=self.scale_factor,
            mode='nearest'
        )

        # Set up the U-Net
        self.unet = ConvUNet(
            input_shape=(out_H, out_W, in_C + self.out_features),
            output_shape=(out_H, out_W, out_C),
            scale=3,
            num_filters=32
        )

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)

        # shift into [-1, 1]
        out = input
        out = (out * 2.0) - 1.0

        # Feature extract
        #print(out.shape)
        out = self.feat_extract.forward(out)

        # upsample
        #print(out.shape)
        out = self.upsampling(out)

        # U-net

        # encode
        #print(out.shape)
        out, skip = self.unet.encoder.forward(out)

        # decode
        #print(out.shape)
        out = self.unet.decoder(out, skip)

        #print(out.shape)
        return out

    def loss(self, in_x, target_x):
        in_x = in_x.permute(0, 3, 1, 2)
        target_x = target_x.permute(0, 3, 1, 2)

        # shift to [-1, 1]
        out = in_x
        out = (out * 2.0) - 1.0

        # Feature extract
        out = self.feat_extract.forward(out)

        # upsample
        out = self.upsampling(out)

        # U-net (avoid the scaling in the other network so do this here)

        # encode
        out, skip = self.unet.encoder.forward(out)

        # decode
        out = self.unet.decoder(out, skip)

        loss = 1.0 - self.unet.ssim_loss.forward(out, target_x)

        return loss

    def loss_with_frame(self, frameObject, targetFrameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npTargetFrameBuffer = targetFrameObject.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer)
        torchTargetImageBuffer = torchTargetImageBuffer.unsqueeze(0)

        # Run the model
        torchLoss = self.loss(
            torchImageBuffer, torchTargetImageBuffer
        )

        # return an image
        return torchLoss

    def forward_with_frame(self, frameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0)

        # Run the model (squeeze, permute and shift)
        torchOutput = self.forward(torchImageBuffer)
        #torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        torchOutput = torchOutput.squeeze().permute(1, 2, 0)

        # return an image
        return image(torchBuffer=torchOutput)


