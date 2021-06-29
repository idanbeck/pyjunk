import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, input_rgb_shape, input_depth_shape, output_shape, scale=4, num_filters=32,
                 ssim_window_size=11, lambda_vgg=0.1, prob_aug_noise=0.8, lambda_augment=0.001,
                 upsample_mode='nearest', fZeroSampling=True, fLearnedMask=False, fAugmentNoise=True,
                 *args, **kwargs):

        # input shape is h, w, c
        self.input_rgb_shape = input_rgb_shape
        self.input_depth_shape = input_depth_shape
        self.output_shape = output_shape
        self.scale = scale      # Note sale is per-axis scaling
        self.num_filters = num_filters
        self.out_features = 8
        self.num_feat_filters = 32
        self.fAugmentNoise = fAugmentNoise
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

    def yCbCr2rgb(self, input_im):
        B, C, H, W = input_im.shape

        input_im = input_im.permute(0, 2, 3, 1)
        input_shape = input_im.shape

        #im_flat = input_im.contiguous().view(-1, 3).float()
        im_flat = input_im.view(-1, 3)

        mat = torch.tensor([[1.164, 1.164, 1.164],
                            [0, -0.392, 2.017],
                            [1.596, -0.813, 0]])

        bias = torch.tensor([-16.0 / 255.0, -128.0 / 255.0, -128.0 / 255.0])

        temp = (im_flat + bias).mm(mat)
        out = temp.view(input_shape)

        out = out.permute(0, 3, 1, 2)

        return out

    def rgb2yCbCr(self, input_im):
        B, C, H, W = input_im.shape

        input_im = input_im.permute(0, 2, 3, 1)
        input_shape = input_im.shape

        #im_flat = input_im.contiguous().view(-1, 3).float()
        im_flat = input_im.view(-1, 3)
        mat = torch.tensor([[0.257, -0.148, 0.439],
                            [0.564, -0.291, -0.368],
                            [0.098, 0.439, -0.071]])
        bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])

        temp = im_flat.mm(mat) + bias
        out = temp.view(input_shape)

        out = out.permute(0, 3, 1, 2)

        return out

    def forward(self, in_x):
        strType = 'rgbd'

        if(self.params_for_forward != None):
            strType = self.params_for_forward.get('type', 'rgbd')

        handlers = {
            'rgbd': lambda in_rgbd : self.forward_rgbtoycbcr(in_rgbd),
            'ycbcrd': lambda in_ycbcrd: self.forward_onnx(in_ycbcrd),
        }

        return handlers[strType](in_x)

    # Expects input in NHWC format
    def forward_rgbtoycbcr(self, in_rgbd):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_B, in_H, in_W, in_C = in_rgbd.shape

        input_rgb = in_rgbd[:, :, :, 0:3]

        if(in_depth_C == 1):
            input_depth = in_rgbd[:, :, :, 3].unsqueeze(dim=3)
        else:
            input_depth = in_rgbd[:, :, :, 3:(3+in_depth_C)]

        input_depth = input_depth.permute(0, 3, 1, 2)
        input_rgb = input_rgb.permute(0, 3, 1, 2)

        # RGB -> YCbCr
        input_ycbcr = ycbcr(input_rgb)

        out_input = torch.cat([input_ycbcr, input_depth], dim=1)
        out_input_scaled = (out_input * 2.0) - 1.0  # shift into [-1, 1]

        # Feature extract
        out_feat = self.feat_extract.forward(out_input_scaled)

        # upsample
        out_upsampled = self.upsampling(out_feat)

        if (self.fZeroSampling == True):
            out_zerosampled = out_upsampled * self.zero_upsampling_mask[:, :, :in_H * self.scale_factor, :in_W * self.scale_factor]

        # U-net

        # encode
        out_unet_enc, skip = self.unet.encoder.forward(out_zerosampled)

        # decode
        out_unet_dec = self.unet.decoder(out_unet_enc, skip)

        # Convert back to rgb for output
        out_ycbcr = out_unet_dec * 0.5 + 0.5
        out_rgb = rgb(out_ycbcr)

        # switch back to NHWC format (for onnx)
        out_rgb_permuted = out_rgb.permute(0, 2, 3, 1)
        out = out_rgb_permuted

        return out

    # This path assumes the input is in ycbcrd
    # format already and also outputs the YCbCr output
    # This is used for the ONNX path for the Barracuda pipeline in Unity
    # Expects input in NCHW format
    def forward_onnx(self, in_ycbcrd):

        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_B, in_H, in_W, in_C = in_ycbcrd.shape

        input_ycbcr = in_ycbcrd[:, :3, :, :]
        input_depth = in_ycbcrd[:, 3:(3 + in_depth_C), :, :]

        out_input = torch.cat([input_ycbcr, input_depth], dim=1)
        out_input_scaled = (out_input * 2.0) - 1.0  # shift into [-1, 1]

        # Feature extract
        out_feat = self.feat_extract.forward(out_input_scaled)

        # upsample
        out_upsampled = self.upsampling(out_feat)

        if(self.fZeroSampling == True):
           out_zerosampled = out_upsampled * self.zero_upsampling_mask[:, :, :in_H * self.scale_factor, :in_W * self.scale_factor]

        # U-net

        # encode
        out_unet_enc, skip = self.unet.encoder.forward(out_zerosampled)

        # decode
        out_unet_dec = self.unet.decoder(out_unet_enc, skip)

        # Convert back to rgb for output
        out_ycbcr = out_unet_dec * 0.5 + 0.5
        return out_ycbcr

    def loss(self, in_x, target_x):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        B, C, H, W = target_x.shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape
        in_B, in_H, in_W, in_C = in_x.shape

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
            #out = out * self.zero_upsampling_mask
            out = out * self.zero_upsampling_mask[:, :, :in_H * self.scale_factor, :in_W * self.scale_factor]

        # U-net (avoid the scaling in the other network so do this here)

        # encode
        out, skip = self.unet.encoder.forward(out)

        # decode
        out = self.unet.decoder(out, skip)
        out = out * 0.5 + 0.5

        # loss = 1.0 - self.unet.ssim_loss.forward(out, target_x)
        # loss = (1.0 - self.unet.ssim_loss.forward(out, target_x)) + 0.25 * torch.norm(out - target_x, 1)*(1.0/(C * H * W))

        # ssim_loss = 0.5 * (1.0 - kornia.losses.ssim(out, target_x_ycbcr, 11))
        #ssim_loss = (1.0 - kornia.losses.ssim(out, target_x_ycbcr, self.ssim_window_size))
        ssim_loss = (1.0 - kornia.losses.ssim(target_x_ycbcr, out, self.ssim_window_size))
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

        # This is for sure a hack
        if (in_depth_C > 3):
            in_depth_C = 3
        if (in_rgb_C > 3):
            in_rgb_C = 3

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

    def loss_with_frames(self, sourceFrames, targetFrames, lr_patch_size=None, num_patches=None):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape
        out_rgb_H, out_rgb_W, out_rgb_C = self.output_shape
        scale_factor = out_rgb_H // in_rgb_H

        hr_patch_size = None
        if(lr_patch_size != None):
            hr_patch_size = (lr_patch_size[0] * scale_factor, lr_patch_size[1] * scale_factor)

        # This is for sure a hack
        if (in_depth_C > 3):
            in_depth_C = 3
        if (in_rgb_C > 3):
            in_rgb_C = 3

        pbar = tqdm_notebook(zip(sourceFrames, targetFrames), desc='loading frame', leave=False,
                             total=len(sourceFrames))

        x_lr = None
        x_hr = None

        for frame_lr, frame_hr in pbar:
            patch_extents = []

            if(num_patches != None and lr_patch_size != None and hr_patch_size != None):
                # print(frame_lr.strFrameID)
                # print(frame_hr.strFrameID)

                for patch in range(num_patches):
                    # Create N LR/HR patches
                    lrH, lrW = lr_patch_size

                    minLRY, maxLRY = 0, in_rgb_H - lrH
                    minLRX, maxLRX = 0, in_rgb_W - lrW

                    lrY = np.random.randint(minLRY, maxLRY)
                    lrX = np.random.randint(minLRX, maxLRX)

                    lr_patch_extents = (lrY, lrX, lrH, lrW)
                    hr_patch_extents = (lrY * scale_factor, lrX * scale_factor, lrH * scale_factor, lrW * scale_factor)

                    patch_extents.append((lr_patch_extents, hr_patch_extents))

            #print(patch_extents)

            npFrameLRBuffer = frame_lr.GetNumpyBuffer()
            torchImageLRBuffer = torch.FloatTensor(npFrameLRBuffer)
            torchImageLRBuffer = torchImageLRBuffer.unsqueeze(0).to(ptu.GetDevice())
            torchImageLRBuffer = torchImageLRBuffer[:, :, :, :(in_rgb_C + in_depth_C)]
            # torchImageLRBuffer = torchImageLRBuffer.permute(0, 3, 1, 2)

            x_lr_ = None
            if(len(patch_extents) > 0):
                for lr_patch_extent, _ in patch_extents:
                    Y, X, H, W = lr_patch_extent
                    startX, endX = X, X + W
                    startY, endY = Y, Y + H
                    lr_patch = torchImageLRBuffer[:, startY:endY, startX:endX, :]
                    lr_patch = lr_patch.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                    # print("lr_patch")
                    # print(lr_patch.shape)

                    if(x_lr_ == None):
                        x_lr_ = lr_patch
                    else:
                        x_lr_ = torch.cat((x_lr_, lr_patch), dim=0)

                    # print("x_lr_")
                    # print(x_lr_.shape)
            else:
                x_lr_ = torchImageLRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0

            if (x_lr == None):
                x_lr = x_lr_
            else:
                x_lr = torch.cat((x_lr, x_lr_), dim=0)

            # print("x_lr")
            # print(x_lr.shape)

            npFrameHRBuffer = frame_hr.GetNumpyBuffer()
            torchImageHRBuffer = torch.FloatTensor(npFrameHRBuffer)
            torchImageHRBuffer = torchImageHRBuffer.unsqueeze(0).to(ptu.GetDevice())
            torchImageHRBuffer = torchImageHRBuffer[:, :, :, :3]  # bit of a hack tho
            # torchImageHRBuffer = torchImageHRBuffer.permute(0, 3, 1, 2)

            x_hr_ = None
            if (len(patch_extents) > 0):
                for _, hr_patch_extent in patch_extents:
                    Y, X, H, W = hr_patch_extent
                    startX, endX = X, X + W
                    startY, endY = Y, Y + H
                    hr_patch = torchImageHRBuffer[:, startY:endY, startX:endX, :]
                    hr_patch = hr_patch.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                    if (x_hr_ == None):
                        x_hr_ = hr_patch
                    else:
                        x_hr_ = torch.cat((x_hr_, hr_patch), dim=0)
            else:
                x_hr_ = torchImageHRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0

            if (x_hr == None):
                x_hr = x_hr_
            else:
                x_hr = torch.cat((x_hr, x_hr_), dim=0)

            # print("x_hr")
            # print(x_hr.shape)

        # Run the model
        torchLoss = self.loss(
            torchImageLRBuffer, torchImageHRBuffer
        )

        # return an image
        return torchLoss

    def forward_with_frame(self, frameObject, lr_patch_size=None):
        in_rgb_H, in_rgb_W, in_rgb_C = self.input_rgb_shape
        in_depth_H, in_depth_W, in_depth_C = self.input_depth_shape

        # This is for sure a hack
        if (in_depth_C > 3):
            in_depth_C = 3
        if (in_rgb_C > 3):
            in_rgb_C = 3

        patch_extents = None
        if(lr_patch_size != None):
            patchH, patchW = lr_patch_size
            minY, maxY = 0, in_rgb_H - patchH
            minX, maxX = 0, in_rgb_W - patchW
            Y = np.random.randint(minY, maxY)
            X = np.random.randint(minX, maxX)
            patch_extents = (Y, X, patchH, patchW)

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameBuffer = frameObject.GetNumpyBuffer(patch_extents=patch_extents)
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())
        torchImageBuffer = torchImageBuffer[:, :, :, :(in_rgb_C + in_depth_C)]

        # Run the model (squeeze, permute and shift)
        torchOutput_raw = self.forward(torchImageBuffer)

        # torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        #torchOutput = torchOutput.squeeze().permute(1, 2, 0)
        torchOutput = torchOutput_raw.squeeze()

        # return an image
        return image(torchBuffer=torchOutput), torchOutput_raw


