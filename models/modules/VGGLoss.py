import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Simple module that allows for calculating the VGG16 perceptual distance

class VGGOutput(object):
    def __init__(self, relu1_2, relu2_2, relu3_3, relu4_3, *args, **kwargs):
        super(VGGOutput, self).__init__(*args, **kwargs)
        self.__dict__ = locals()

class VGGLoss(nn.Module):
    def __init__(self, requires_grad=False, *args, **kwargs):
        super(VGGLoss, self).__init__(*args, **kwargs)
        vgg_pretrained_feats = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.requires_grad = requires_grad

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_feats[x])

        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_feats[x])

        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_feats[x])

        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_feats[x])

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        out = input

        out = self.slice1(out)
        h_relu1_2 = out

        out = self.slice2(out)
        h_relu2_2 = out

        out = self.slice3(out)
        h_relu3_3 = out

        out = self.slice4(out)
        h_relu4_3 = out

        #return VGGOutput(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

    def loss(self, x, y):
        vgg_out_x = self.forward(x)
        vgg_out_y = self.forward(y)

        loss = 0

        for h_x, h_y in zip(vgg_out_x, vgg_out_y):
            loss += F.l1_loss(h_x, h_y)

        #loss /= len(vgg_out_x)

        return loss