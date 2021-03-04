import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import numpy as np

from repos.pyjunk.junktools import utils

# Base Solver class

class TorchSolver():
    def __init__(self, model, params, *args, **kwargs):
        super(TorchSolver, self).__init__(*args, *kwargs)
        self.params = params
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.grad_clip = params.get('grad_clip')    # If None won't do anything below
        self.model = model
        self.batch_size = params.get('batch_size')
        self.test_batch_size = params.get('test_batch_size')
        self.betas = params.get('betas')
        self.eps = params.get('eps')

        self.strOptimizer = params['strOptimizer']
        if(self.strOptimizer == 'Adam'):
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr, betas = self.betas, eps = self.eps)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr, betas = self.betas, eps = self.eps)



