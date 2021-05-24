import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
from repos.pyjunk.models.modules.SSIMModule import SSIMModule

import repos.pyjunk.junktools.pytorch_utils as ptu

import math
import kornia # This is mainly for A/B testing SSIM and maybe the YCbCr shit

# Create a stereo Conv U-Net
# The idea is to have two seprate encoders and one shared decoder

# TODO: The encoder / decoders could likely be generalized and used across U-Nets

class StereoConvUNetEncoder(nn.Module):
    def __init__(self, input_shape, scale=3, num_filters=32, *args, **kwargs):
        super(StereoConvUNetEncoder, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.scale = scale

        # TODO: Generalize pooling

        self.net = []

        # input shape is h, w, c
        H, W, C = input_shape

        print(C)

        # First module
        print("enc first: %d - %d" % (C, self.num_filters))
        first_module = [
            nn.Conv2d(C, self.num_filters * 2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_filters * 2, self.num_filters, 3, 1, 1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
        ]
        # first_module = nn.ModuleList(*[first_module])
        # self.modules.append(first_module)
        self.net.extend(*[first_module])

        for k in range(1, self.scale - 1):
            print("enc inner: %d - %d" % (self.num_filters * (2 ** (k - 1)), self.num_filters * (2 ** k)))
            next_module = [
                nn.Conv2d(self.num_filters * (2 ** (k - 1)),
                          self.num_filters * (2 ** k), 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(self.num_filters * (2 ** k),
                          self.num_filters * (2 ** k), 3, 1, 1),
                nn.LeakyReLU(),
                nn.AvgPool2d(2),
            ]
            # next_module = nn.ModuleList(*[next_module])
            # self.modules.append(next_module)
            self.net.extend(*[next_module])

        # Final module
        print("enc final: %d - %d" % (self.num_filters * (2 ** (scale - 2)), self.num_filters * (2 ** (scale - 1))))
        last_module = [
            nn.Conv2d(self.num_filters * (2 ** (self.scale - 2)),
                      self.num_filters * (2 ** (self.scale - 1)), 3, 1, 1),
            nn.LeakyReLU(),
        ]
        # last_module = nn.ModuleList(*[last_module])
        # self.modules.append(last_module)
        self.net.extend(*[last_module])

        #self.modules = nn.ModuleList(*[self.modules])
        self.net = nn.ModuleList(*[self.net])

    def forward(self, input):
        out = input
        module_outputs = []

        for layer in self.net:
            #print(out.shape)
            # Grab the output right before it goes to the pooling layer
            if(isinstance(layer, nn.AvgPool2d)):
                #print(out.shape)
                module_outputs.append(out)
            out = layer(out)

        #print(out.shape)
        return out, module_outputs

class StereoConvUNetDecoder(nn.Module):
    def __init__(self, output_shape, scale=3, num_filters=32, *args, **kwargs):
        super(StereoConvUNetDecoder, self).__init__(*args, **kwargs)
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters

        # TODO: Generalize upsampling

        self.net = []

        # output shape is h, w, c
        H, W, C = output_shape

        scale = self.scale

        # First module
        in_c = self.num_filters * (2 ** (self.scale - 1)) * 2  # times two since we concat the stereo inputs
        out_c = self.num_filters * (2 ** (self.scale - 2))
        print("dec first: %d - %d" % (in_c, out_c))
        first_module = [
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ]
        # first_module = nn.ModuleList(*[first_module])
        # self.modules.append(first_module)
        self.net.extend(*[first_module])

        for k in reversed(range(2, self.scale)):
            in_c = self.num_filters * ((2 ** k) + 2) # times two because of the stereo skip connections
            inner_c = self.num_filters * (2 ** (k - 1))
            out_c = self.num_filters * (2 ** (k - 2))
            print("dec inner: %d - %d - %d" % (in_c, inner_c, out_c))
            next_module = [
                nn.Conv2d(in_c, inner_c, 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(inner_c, out_c, 3, 1, 1),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ]
            # next_module = nn.ModuleList(*[next_module])
            # self.modules.append(next_module)
            self.net.extend(*[next_module])

        # Last module
        in_c = self.num_filters * (2 + 1)
        inner_c = self.num_filters
        out_c = C
        print("dec final: %d - %d - %d" % (in_c, inner_c, out_c))
        last_module = [
            nn.Conv2d(in_c, inner_c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(inner_c, out_c, 3, 1, 1),
            #nn.ReLU(),
            nn.Tanh()
            #nn.Sigmoid()
        ]
        # last_module = nn.ModuleList(*[last_module])
        # self.modules.append(last_module)
        self.net.extend(*[last_module])

        #self.modules = nn.ModuleList(*[self.modules])
        self.net = nn.ModuleList(*[self.net])

    def forward(self, in_left, in_right, skip_connections_left, skip_connections_right):
        out_left = in_left
        out_right = in_right

        # Concatenate skip connections
        skip_id = len(skip_connections_left) - 1
        fFirst = True

        #print(in_left.shape)
        #print(in_right.shape)

        out = torch.cat((in_left, in_right), dim=1).to(ptu.GetDevice())

        for layer in self.net:
            #print("yo")
            #print(out.shape)
            out = layer(out)

            # If we just ran an upsample then plop in the skipperoonie
            if (isinstance(layer, nn.Upsample)):
                out = torch.cat((skip_connections_left[skip_id], skip_connections_right[skip_id], out), dim=1).to(ptu.GetDevice())
                skip_id -= 1

        return out

class StereoConvUNet(Model):
    def __init__(self, input_shape, output_shape, scale=3, num_filters=32, channels=3, *args, **kwargs):
        # input shape is h, w, c
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters
        self.channels = channels
        super(StereoConvUNet, self).__init__(*args, **kwargs)

    def ConstructModel(self):
        H, W, C = self.input_shape

        # TODO: Fix this
        # self.ssim_loss = SSIMModule(
        #     window_size=11,
        #     sigma=1.5,
        #     channels=self.channels,
        #     c1=1e-4,
        #     c2=9e-4,
        #     fSizeAverage=True
        # ).to(ptu.GetDevice())

        # Set up the encoder and decoder
        self.left_encoder = StereoConvUNetEncoder(
            input_shape=self.input_shape,
            scale=self.scale,
            num_filters=self.num_filters
        ).to(ptu.GetDevice())

        self.right_encoder = StereoConvUNetEncoder(
            input_shape=self.input_shape,
            scale=self.scale,
            num_filters=self.num_filters
        ).to(ptu.GetDevice())

        self.decoder = StereoConvUNetDecoder(
            output_shape=self.output_shape,
            scale=self.scale,
            num_filters=self.num_filters
        ).to(ptu.GetDevice())

    def forward(self, in_left, in_right):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        in_left = in_left.permute(0, 3, 1, 2)
        in_right = in_right.permute(0, 3, 1, 2)

        # shift into [-1, 1]
        in_left_ycbcr = ycbcr(in_left).to(ptu.GetDevice())
        in_left_ycbcr = (in_left_ycbcr * 2.0) - 1.0

        in_right_ycbcr = ycbcr(in_right).to(ptu.GetDevice())
        in_right_ycbcr = (in_right_ycbcr * 2.0) - 1.0

        # encode
        enc_out_left, skip_connections_left = self.left_encoder.forward(in_left_ycbcr)
        enc_out_right, skip_connections_right = self.right_encoder.forward(in_right_ycbcr)

        # decode
        out = self.decoder.forward(enc_out_left, enc_out_right, skip_connections_left, skip_connections_right)

        # Convert back to rgb for output
        out_rgb= rgb(out * 0.5 + 0.5).to(ptu.GetDevice())

        return out_rgb

    def loss(self, in_left, in_right, target_x):
        ycbcr = kornia.color.RgbToYcbcr()
        rgb = kornia.color.YcbcrToRgb()

        in_left = in_left.permute(0, 3, 1, 2)
        in_left_ycbcr = ycbcr(in_left).to(ptu.GetDevice())
        in_left_ycbcr = (in_left_ycbcr * 2.0) - 1.0         # shift to [-1, 1]

        in_right = in_right.permute(0, 3, 1, 2)
        in_right_ycbcr = ycbcr(in_right).to(ptu.GetDevice())
        in_right_ycbcr = (in_right_ycbcr * 2.0) - 1.0       # shift to [-1, 1]

        target_x_rgb = target_x.permute(0, 3, 1, 2)
        target_x_ycbcr = ycbcr(target_x_rgb).to(ptu.GetDevice())

        # encode
        #print(out.shape)
        enc_out_left, skip_connections_left = self.left_encoder.forward(in_left_ycbcr)
        enc_out_right, skip_connections_right = self.right_encoder.forward(in_right_ycbcr)

        # decode
        #print(out.shape)
        #out = self.decoder(out, skip)
        out = self.decoder.forward(enc_out_left, enc_out_right, skip_connections_left, skip_connections_right)
        out = out * 0.5 + 0.5
        #out = rgb(out * 0.5 + 0.5)

        #print(out.shape)
        #loss = 0.5 * (1.0 - self.ssim_loss.forward(out, target_x))
        loss = (1.0 - kornia.losses.ssim(out, target_x_ycbcr, 11)).to(ptu.GetDevice())
        #loss = loss.mean(1).mean(1).mean(1)
        loss = loss.mean()

        return loss

    def loss_with_frame(self, frameLeftObject, frameRightObject, targetFrameObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameLeftBuffer = frameLeftObject.GetNumpyBuffer()
        torchImageLeftBuffer = torch.FloatTensor(npFrameLeftBuffer)
        torchImageLeftBuffer = torchImageLeftBuffer.unsqueeze(0)
        torchImageLeftBuffer = torchImageLeftBuffer[:, :, :, :3]  # bit of a hack tho

        npFrameRightBuffer = frameRightObject.GetNumpyBuffer()
        torchImageRightBuffer = torch.FloatTensor(npFrameRightBuffer)
        torchImageRightBuffer = torchImageRightBuffer.unsqueeze(0)
        torchImageRightBuffer = torchImageRightBuffer[:, :, :, :3]  # bit of a hack tho

        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npTargetFrameBuffer = targetFrameObject.GetNumpyBuffer()
        torchTargetImageBuffer = torch.FloatTensor(npTargetFrameBuffer)
        torchTargetImageBuffer = torchTargetImageBuffer.unsqueeze(0)
        torchTargetImageBuffer = torchTargetImageBuffer[:, :, :, :3]  # bit of a hack tho

        # Run the model
        torchLoss = self.loss(
            torchImageLeftBuffer, torchImageRightBuffer, torchTargetImageBuffer
        )

        # return an image
        return torchLoss

    def forward_with_frame(self, frameLeftObject, frameRightObject):
        # Grab the torch tensor from the frame (this may be a particularly deep tensor)
        npFrameLeftBuffer = frameLeftObject.GetNumpyBuffer()
        torchImageLeftBuffer = torch.FloatTensor(npFrameLeftBuffer)
        torchImageLeftBuffer = torchImageLeftBuffer.unsqueeze(0)
        torchImageLeftBuffer = torchImageLeftBuffer[:, :, :, :3]  # bit of a hack tho

        npFrameRightBuffer = frameRightObject.GetNumpyBuffer()
        torchImageRightBuffer = torch.FloatTensor(npFrameRightBuffer)
        torchImageRightBuffer = torchImageRightBuffer.unsqueeze(0)
        torchImageRightBuffer = torchImageRightBuffer[:, :, :, :3]  # bit of a hack tho

        # Run the model (squeeze, permute and shift)
        torchOutput = self.forward(torchImageLeftBuffer, torchImageRightBuffer)
        #torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        torchOutput = torchOutput.squeeze().permute(1, 2, 0)

        # return an image
        return image(torchBuffer=torchOutput)


