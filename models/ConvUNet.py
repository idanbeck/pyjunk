import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image
from repos.pyjunk.models.modules.SSIMModule import SSIMModule

import math

class ConvUNetEncoder(nn.Module):
    def __init__(self, input_shape, scale=3, num_filters=32, *args, **kwargs):
        super(ConvUNetEncoder, self).__init__(*args, **kwargs)
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

        # for module in self.modules.all():
        #     #print(out.shape)
        #
        #     for layer in module:
        #         # Grab the output right before it goes to the pooling layer
        #         if(isinstance(layer, nn.AvgPool2d)):
        #             #print(out.shape)
        #             module_outputs.append(out)
        #         out = layer(out)
        #
        #     # Cache module outputs

        for layer in self.net:
            #print(out.shape)
            # Grab the output right before it goes to the pooling layer
            if(isinstance(layer, nn.AvgPool2d)):
                #print(out.shape)
                module_outputs.append(out)
            out = layer(out)

        #print(out.shape)
        return out, module_outputs

class ConvUNetDecoder(nn.Module):
    def __init__(self, output_shape, scale=3, num_filters=32, *args, **kwargs):
        super(ConvUNetDecoder, self).__init__(*args, **kwargs)
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters

        # TODO: Generalize upsampling

        self.net = []

        # output shape is h, w, c
        H, W, C = output_shape

        scale = self.scale

        # First module
        in_c = self.num_filters * (2 ** (self.scale - 1))
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
            in_c = self.num_filters * (2 ** k)
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
        in_c = self.num_filters * 2
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

    def forward(self, input, skip_connections):
        out = input

        # # Concatenate skip connections
        # skip_id = len(skip_connections) - 1
        # fFirst = True
        #
        # for module in self.modules:
        #
        #     # First layer skip connect is just the output so skip it
        #     if(fFirst == False):
        #         # print("%d" % skip_id)
        #         # print(out.shape)
        #         # print(skip_connections[skip_id].shape)
        #         out = torch.cat((skip_connections[skip_id], out), dim=1)
        #         skip_id -= 1
        #     else:
        #         fFirst = False
        #
        #     #print(out.shape)
        #     for layer in module:
        #         out = layer(out)

        # Concatenate skip connections
        skip_id = len(skip_connections) - 1
        fFirst = True

        for layer in self.net:
            out = layer(out)

            # If we just ran an upsample then plop in the skipperoonie
            if (isinstance(layer, nn.Upsample)):
                out = torch.cat((skip_connections[skip_id], out), dim=1)
                skip_id -= 1

        return out

class ConvUNet(Model):
    def __init__(self, input_shape, output_shape, scale=3, num_filters=32, channels=3, *args, **kwargs):
        super(ConvUNet, self).__init__(*args, **kwargs)

        # input shape is h, w, c
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scale = scale
        self.num_filters = num_filters
        self.channels = channels

        H, W, C = input_shape

        self.ssim_loss = SSIMModule(
            window_size=11,
            sigma=1.5,
            channels=self.channels,
            c1=1e-4,
            c2=9e-4,
            fSizeAverage=True
        )

        # Set up the encoder and decoder
        self.encoder = ConvUNetEncoder(
            input_shape=self.input_shape,
            scale=self.scale,
            num_filters=self.num_filters
        )

        self.decoder = ConvUNetDecoder(
            output_shape=self.output_shape,
            scale=self.scale,
            num_filters=self.num_filters
        )

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)

        # shift into [-1, 1]
        out = input
        out = (out * 2.0) - 1.0

        # encode
        out, skip_connections = self.encoder.forward(out)

        # decode
        out = self.decoder.forward(out, skip_connections)

        return out

    def loss(self, in_x, target_x):
        in_x = in_x.permute(0, 3, 1, 2)
        target_x = target_x.permute(0, 3, 1, 2)
        #target_x = (target_x * 2.0) - 1.0

        # shift to [-1, 1]
        out = in_x
        out = (out * 2.0) - 1.0

        # encode
        #print(out.shape)
        out, skip = self.encoder.forward(out)

        # decode
        #print(out.shape)
        out = self.decoder(out, skip)

        #out = out * 0.5 + 0.5

        #print(out.shape)
        loss = 0.5 * (1.0 - self.ssim_loss.forward(out, target_x))

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


