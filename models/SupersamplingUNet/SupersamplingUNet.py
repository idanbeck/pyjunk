import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

from repos.pyjunk.models.ConvUNet import ConvUNet

from repos.pyjunk.models.modules.VGGLoss import VGGLoss

import repos.pyjunk.junktools.pytorch_utils as ptu

import math
import kornia

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
    def __init__(self, input_rgb_shape, input_depth_shape, output_shape, scale=3, num_filters=32, *args, **kwargs):

        # input shape is h, w, c
        self.input_rgb_shape = input_rgb_shape
        self.input_depth_shape = input_depth_shape
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters
        self.out_features = 8
        self.num_feat_filters = 32
        self.fAugmentNoise = True
        self.lambda_augment = 0.001

        super(SupersamplingUNet, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        out_H, out_W, out_C = self.output_shape

        self.scale_factor = out_H // in_rgb_H

        feat_extract_input_shape = (in_rgb_H, in_rgb_W, in_rgb_C + in_depth_C)
        print(feat_extract_input_shape)

        # Feature Extraction Network
        self.feat_extract = SSFeatExtract(
            input_shape=feat_extract_input_shape,
            num_filters=self.num_feat_filters,
            out_features=self.out_features
        ).to(ptu.GetDevice())

        # upsampling (TODO: Zero upsampling implementation)
        self.upsampling = nn.Upsample(
            #size=self.output_shape,
            scale_factor=self.scale_factor,
            mode='nearest'
        ).to(ptu.GetDevice())

        # Set up the U-Net
        self.unet = ConvUNet(
            input_shape=(out_H, out_W, in_rgb_C + in_depth_C + self.out_features),
            output_shape=(out_H, out_W, out_C),
            scale=3,
            num_filters=32
        ).to(ptu.GetDevice())

        self.vgg_loss = VGGLoss(
            requires_grad=False
        ).to(ptu.GetDevice())

        if(self.fAugmentNoise == True):
            self.input_noise = torch.distributions.normal.Normal(
                torch.zeros(feat_extract_input_shape),
                torch.ones(feat_extract_input_shape)
            )

            self.output_noise = torch.distributions.normal.Normal(
                torch.zeros(self.output_shape),
                torch.ones(self.output_shape)
            )

    def forward(self, input):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        input_rgb = input[:, :, :, 0:3]
        input_depth = input[:, :, :, 3].unsqueeze(dim=3)
        input_depth = input_depth.permute(0, 3, 1, 2)
        input_rgb = input_rgb.permute(0, 3, 1, 2)
        input_ycbcr = ycbcr(input_rgb)
        out = torch.cat([input_ycbcr, input_depth], dim=1)
        out = (out * 2.0) - 1.0 # shift into [-1, 1]

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

        # Convert back to rgb for output
        out = rgb(out * 0.5 + 0.5)

        #print(out.shape)
        return out

    def loss(self, in_x, target_x):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        B, C, H, W = target_x.shape

        # in_x = in_x.permute(0, 3, 1, 2)
        # in_x_ycbcr = ycbcr(in_x)
        #
        
        input_rgb = in_x[:, :, :, 0:3]
        input_depth = in_x[:, :, :, 3].unsqueeze(dim=3)
        input_depth = input_depth.permute(0, 3, 1, 2)
        input_rgb = input_rgb.permute(0, 3, 1, 2)
        input_ycbcr = ycbcr(input_rgb)

        #print(target_x.shape)
        # TODO: not sure why we need this, but sometimes we get a non 3 channel input
        if(target_x.shape[3] > 3):
            target_x = target_x[:, :, :, 0:3]

        if (self.fAugmentNoise and torch.rand(1) > 0.5):
            # input_noise = self.input_noise.sample([B])
            # print(input_noise.shape)
            # in_x += self.lambda_augment * self.input_noise.sample([in_x.shape[0]]).to(ptu.GetDevice())
            target_x += self.lambda_augment * self.output_noise.sample([target_x.shape[0]]).to(ptu.GetDevice())

        target_x_rgb = target_x.permute(0, 3, 1, 2)
        target_x_ycbcr = ycbcr(target_x_rgb)

        # shift to [-1, 1]
        # out = in_x_ycbcr
        # out = (out * 2.0) - 1.0
        out = torch.cat([input_ycbcr, input_depth], dim=1)
        out = (out * 2.0) - 1.0  # shift into [-1, 1]

        # Feature extract
        out = self.feat_extract.forward(out)

        # upsample
        out = self.upsampling(out)

        # U-net (avoid the scaling in the other network so do this here)

        # encode
        out, skip = self.unet.encoder.forward(out)

        # decode
        out = self.unet.decoder(out, skip)
        out = out * 0.5 + 0.5

        #loss = 1.0 - self.unet.ssim_loss.forward(out, target_x)
        #loss = (1.0 - self.unet.ssim_loss.forward(out, target_x)) + 0.25 * torch.norm(out - target_x, 1)*(1.0/(C * H * W))

        #ssim_loss = 0.5 * (1.0 - kornia.losses.ssim(out, target_x_ycbcr, 11))
        ssim_loss = (1.0 - kornia.losses.ssim(out, target_x_ycbcr, 11))
        ssim_loss = ssim_loss.mean(1).mean(1).mean(1)

        out_rgb = rgb(out)
        vgg_loss = self.vgg_loss.loss(out_rgb, target_x_rgb)
        # print(vgg_loss)

        loss = ssim_loss + vgg_loss
        #loss = ssim_loss

        return loss

    def loss_with_frame(self, frameObject, targetFrameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npTargetFrameBuffer = targetFrameObject.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer)
        torchTargetImageBuffer = torchTargetImageBuffer.unsqueeze(0).to(ptu.GetDevice())

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
        torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())

        # Run the model (squeeze, permute and shift)
        torchOutput = self.forward(torchImageBuffer)
        #torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        torchOutput = torchOutput.squeeze().permute(1, 2, 0)

        # return an image
        return image(torchBuffer=torchOutput)


