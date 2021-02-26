import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

from repos.pyjunk.models.GQN.TowerRepresentationNetwork import TowerRepresentationNetwork
from repos.pyjunk.models.GQN.PyramidRepresentationNetwork import PyramidRepresentationNetwork

# Generative Query Network Model

class GQNModel(Model):
    def __init__(self, input_shape, *args, **kwargs):
        super(GQNModel, self).__init__(*args, **kwargs)

        self.input_shape = input_shape

    def loss(self, input):
        pass

