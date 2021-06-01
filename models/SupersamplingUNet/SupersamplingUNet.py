import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

from repos.pyjunk.models.ConvUNet import ConvUNet

from repos.pyjunk.models.modules.VGGLoss import VGGLoss

import repos.pyjunk.junktools.pytorch_utils as ptu

from tqdm import trange, tqdm_notebook

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
    def __init__(self, input_rgb_shape, input_depth_shape, output_shape, scale=3, num_filters=32,
                 ssim_window_size=11, lambda_vgg=0.1, prob_aug_noise=0.8, lambda_augment=0.001,
                 upsample_mode='nearest', fZeroSampling=True, fLearnedMask=False, *args, **kwargs):

        # input shape is h, w, c
        self.input_rgb_shape = input_rgb_shape
        self.input_depth_shape = input_depth_shape
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters
        self.out_features = 8
        self.num_feat_filters = 32
        self.fAugmentNoise = True
        self.lambda_augment = lambda_augment
        self.ssim_window_size = ssim_window_size
        self.lambda_vgg = lambda_vgg
        self.prob_aug_noise = prob_aug_noise
        self.upsample_mode = upsample_mode
        self.fZeroSampling = fZeroSampling
        self.fLearnedMask = fLearnedMask

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
            # size=self.output_shape,
            scale_factor=self.scale_factor,
            mode=self.upsample_mode
        ).to(ptu.GetDevice())

        if(self.fZeroSampling):
            #self.zero_upsampling_mask = torch.autograd.Variable(ptu.zeros(1, 1, out_H, out_W), requires_grad=True)
            #self.zero_upsampling_mask = nn.Parameter(ptu.zeros(1, 1, out_H, out_W), requires_grad=True)
            if(self.fLearnedMask):
                self.zero_upsampling_mask = nn.Parameter(ptu.zeros(1, 1, out_H, out_W), requires_grad=True)
                #nn.init.sparse_(self.zero_upsampling_mask.data, sparsity=0.1)
                nn.init.xavier_normal_(self.zero_upsampling_mask.data)
            else:
                self.zero_upsampling_mask = nn.Parameter(ptu.zeros(1, 1, out_H, out_W), requires_grad=False)
               
                #self.zero_upsampling_mask = ptu.zeros(1, 1, out_H, out_W)
                for i in range(in_rgb_H):
                    for j in range(in_rgb_H):
                        self.zero_upsampling_mask.data[:, :,
                        #self.zero_upsampling_mask[:, :,
                            i * self.scale_factor + self.scale_factor // 2,
                            j * self.scale_factor + self.scale_factor // 2] = 1
                # print(self.zero_upsampling_mask.shape)
                # print(self.zero_upsampling_mask[0, 0, 0, 0:100])

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

        if (self.fAugmentNoise == True):
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

        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        input_rgb = input[:, :, :, 0:3]
        if(in_depth_C == 1):
            input_depth = input[:, :, :, 3].unsqueeze(dim=3)
        else:
            input_depth = input[:, :, :, 3:(3+in_depth_C)]
        input_depth = input_depth.permute(0, 3, 1, 2)
        input_rgb = input_rgb.permute(0, 3, 1, 2)
        input_ycbcr = ycbcr(input_rgb)
        out = torch.cat([input_ycbcr, input_depth], dim=1)
        out = (out * 2.0) - 1.0  # shift into [-1, 1]

        # Feature extract
        # print(out.shape)
        out = self.feat_extract.forward(out)

        # upsample
        # print(out.shape)
        out = self.upsampling(out)

        if(self.fZeroSampling == True):
            out = out * self.zero_upsampling_mask

        # U-net

        # encode
        # print(out.shape)
        out, skip = self.unet.encoder.forward(out)

        # decode
        # print(out.shape)
        out = self.unet.decoder(out, skip)

        # Convert back to rgb for output
        out = rgb(out * 0.5 + 0.5)

        # print(out.shape)
        return out

    def loss(self, in_x, target_x):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        B, C, H, W = target_x.shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        # in_x = in_x.permute(0, 3, 1, 2)
        # in_x_ycbcr = ycbcr(in_x)
        #

        input_rgb = in_x[:, :, :, 0:3]
        #input_depth = in_x[:, :, :, 3].unsqueeze(dim=3)
        if (in_depth_C == 1):
            input_depth = in_x[:, :, :, 3].unsqueeze(dim=3)
        else:
            input_depth = in_x[:, :, :, 3:(3 + in_depth_C)]
        input_depth = input_depth.permute(0, 3, 1, 2)
        input_rgb = input_rgb.permute(0, 3, 1, 2)
        input_ycbcr = ycbcr(input_rgb)

        # print(target_x.shape)
        # TODO: not sure why we need this, but sometimes we get a non 3 channel input
        if (target_x.shape[3] > 3):
            target_x = target_x[:, :, :, 0:3]

        if (self.fAugmentNoise and torch.rand(1) > 1.0 - self.prob_aug_noise):
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

        if (self.fZeroSampling == True):
            # print(out.shape)
            # print(self.zero_upsampling_mask.shape)
            out = out * self.zero_upsampling_mask

        # U-net (avoid the scaling in the other network so do this here)

        # encode
        out, skip = self.unet.encoder.forward(out)

        # decode
        out = self.unet.decoder(out, skip)
        out = out * 0.5 + 0.5

        # loss = 1.0 - self.unet.ssim_loss.forward(out, target_x)
        # loss = (1.0 - self.unet.ssim_loss.forward(out, target_x)) + 0.25 * torch.norm(out - target_x, 1)*(1.0/(C * H * W))

        # ssim_loss = 0.5 * (1.0 - kornia.losses.ssim(out, target_x_ycbcr, 11))
        ssim_loss = (1.0 - kornia.losses.ssim(out, target_x_ycbcr, self.ssim_window_size))
        ssim_loss = ssim_loss.mean(1).mean(1).mean(1)

        out_rgb = rgb(out)
        vgg_loss = self.vgg_loss.loss(out_rgb, target_x_rgb)
        # print(vgg_loss)

        loss = ssim_loss + self.lambda_vgg * vgg_loss
        # loss = ssim_loss

        return loss

    def loss_with_frame(self, frameObject, targetFrameObject):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())
        torchImageBuffer = torchImageBuffer[:, :, :, :(in_rgb_C + in_depth_C)]

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npTargetFrameBuffer = targetFrameObject.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer)
        torchTargetImageBuffer = torchTargetImageBuffer.unsqueeze(0).to(ptu.GetDevice())
        torchTargetImageBuffer = torchTargetImageBuffer[:, :, :, :3]  # bit of a hack tho

        # Run the model
        torchLoss = self.loss(
            torchImageBuffer, torchTargetImageBuffer
        )

        # return an image
        return torchLoss

    def loss_with_frames(self, sourceFrames, targetFrames):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        pbar = tqdm_notebook(zip(sourceFrames, targetFrames), desc='loading frame', leave=False,
                             total=len(sourceFrames))

        x_lr = None
        x_hr = None

        for frame_lr, frame_hr in pbar:
            npFrameLRBuffer = frame_lr.GetNumpyBuffer()
            torchImageLRBuffer = torch.FloatTensor(npFrameLRBuffer)
            torchImageLRBuffer = torchImageLRBuffer.unsqueeze(0).to(ptu.GetDevice())
            torchImageBuffer = torchImageLRBuffer[:, :, :, :(in_rgb_C + in_depth_C)]
            # torchImageLRBuffer = torchImageLRBuffer.permute(0, 3, 1, 2)

            x_lr_ = torchImageLRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0

            if (x_lr == None):
                x_lr = x_lr_
            else:
                x_lr = torch.cat((x_lr, x_lr_), dim=0)

            npFrameHRBuffer = frame_hr.GetNumpyBuffer()
            torchImageHRBuffer = torch.FloatTensor(npFrameHRBuffer)
            torchImageHRBuffer = torchImageHRBuffer.unsqueeze(0).to(ptu.GetDevice())
            torchImageHRBuffer = torchImageHRBuffer[:, :, :, :3]  # bit of a hack tho
            # torchImageHRBuffer = torchImageHRBuffer.permute(0, 3, 1, 2)

            x_hr_ = torchImageHRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
            if (x_hr == None):
                x_hr = x_hr_
            else:
                x_hr = torch.cat((x_hr, x_hr_), dim=0)

        # Run the model
        torchLoss = self.loss(
            torchImageLRBuffer, torchImageHRBuffer
        )

        # return an image
        return torchLoss

    def forward_with_frame(self, frameObject):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())
        torchImageBuffer = torchImageBuffer[:, :, :, :(in_rgb_C + in_depth_C)]

        # Run the model (squeeze, permute and shift)
        torchOutput = self.forward(torchImageBuffer)
        # torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        torchOutput = torchOutput.squeeze().permute(1, 2, 0)

        # return an image
        return image(torchBuffer=torchOutput)


