import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import numpy as np

from repos.pyjunk.junktools import utils
from repos.pyjunk.solvers.Schedulers.AnnealingScheduler import AnnealingScheduler
import repos.pyjunk.junktools.pytorch_utils as ptu

# Base Solver class

class TorchSolver():
    def __init__(self, model, params, *args, **kwargs):
        super(TorchSolver, self).__init__(*args, *kwargs)
        self.params = params
        self.lr = params.get('lr', 0.001)
        self.epochs = params['epochs']
        self.grad_clip = params.get('grad_clip')    # If None won't do anything below
        self.model = model
        self.batch_size = params.get('batch_size')
        self.test_batch_size = params.get('test_batch_size')
        self.betas = params.get('betas', (0.9, 0.999))
        self.eps = params.get('eps', 1e-08)

        self.scheduler_params = params.get('scheduler')

        # Check point settings
        self.checkpoint_file_name = params.get('checkpoint_file_name', None)
        self.checkpoint_epochs = params.get('checkpoint_epochs', 10)
        self.save_test_file_name = params.get('save_test_file_name', None)


        self.strOptimizer = params['strOptimizer']
        if(self.strOptimizer == 'Adam'):
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, betas = self.betas, eps = self.eps)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, betas = self.betas, eps = self.eps)

        if (self.scheduler_params != None):
            strScheduler = self.scheduler_params.get('type')
            mu_i = self.scheduler_params.get('mu_i')
            mu_f = self.scheduler_params.get('mu_f')
            n = self.scheduler_params.get('n')

            if(strScheduler == "Annealing"):
                self.scheduler = AnnealingScheduler(
                    optimizer = self.optimizer,
                    mu_i = mu_i,
                    mu_f = mu_f,
                    n = n
                )
        else:
            self.scheduler = None

    def SaveCheckpoint(self, strCheckpointFilename, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, strCheckpointFilename)

    def LoadCheckpoint(self, strCheckpointFilename):
        if ptu.GetDevice() is None:
            torch_device = torch.device('cpu')
        else:
            torch_device = ptu.GetDevice()
        checkpoint = torch.load(strCheckpointFilename, map_location=torch_device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return epoch, loss

    def SaveONNXCheckpoint(self,
                           strCheckpointFilename,
                           sampleModelInput,
                           opset_version=9,
                           dynamic_axes={},
                           input_names=['input'],
                           output_names=['out'],
                           example_outputs=None,
                           params_for_forward=None
                           ):

        if(params_for_forward != None):
            self.model.params_for_forward = params_for_forward

        # Export the model
        torch.onnx.export(self.model,                   # model being run
                          sampleModelInput,             # model input (or a tuple for multiple inputs)
                          strCheckpointFilename,        # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=opset_version,              # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          input_names=input_names,            # the model's input names
                          output_names=output_names,            # the model's output names
                          verbose=True,
                          dynamic_axes=dynamic_axes
                          )

        self.model.params_for_forward = None


