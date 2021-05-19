import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torch.utils.data as data

import numpy as np
import random

from tqdm import trange, tqdm_notebook

from repos.pyjunk.junktools import utils
import repos.pyjunk.junktools.pytorch_utils  as ptu

#from repos.pyjunk.solvers.TorchSolver import TorchSolver

# SGAN Solver class

class SGANTorchSolver():
    def __init__(self, model, params, *args, **kwargs):
        super(SGANTorchSolver, self).__init__(*args, *kwargs)
        self.model = model
        self.params = params
        self.lr = params.get('lr', 0.001)
        self.epochs = params['epochs']
        self.grad_clip = params.get('grad_clip')  # If None won't do anything below
        self.weight_clip = params.get('weight_clip')
        self.batch_size = params.get('batch_size')
        self.test_batch_size = params.get('test_batch_size')
        self.betas = params.get('betas', (0.5, 0.999))
        self.eps = params.get('eps', 1e-08)
        self.n_critic = params.get('n_critic', 4)

        #self.scheduler_params = params.get('scheduler')

        # Check point settings
        self.checkpoint_file_name = params.get('checkpoint_file_name', None)
        self.checkpoint_epochs = params.get('checkpoint_epochs', 10)
        self.save_test_file_name = params.get('save_test_file_name', None)

        self.strOptimizer = params.get('strOptimizer', 'Adam')

        epoch_lambda = lambda epoch: (self.epochs - epoch) / self.epochs
        self.model.SetupGANOptimizers(solver=self)
        self.model.SetupGANSchedulers(solver=self)


    def SaveCheckpoint(self, strCheckpointFilename, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            # 'generator_optimizer_state_dict': self.model.generator_optimizer.state_dict(),
            # 'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            # 'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
        }, strCheckpointFilename)

    def LoadCheckpoint(self, strCheckpointFilename):
        checkpoint = torch.load(strCheckpointFilename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        # self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        # self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])

        epoch = checkpoint['epoch']

        return epoch

    def train_gan_epochs_frameset(self, train_frameset, fVerbose=False):
        training_losses = []
        self.current_iteration = 0
        g_loss = 0

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)
        for epoch in pbar:
            self.model.generator.train()
            self.model.discriminator.train()
            self.batch_loss_history = []

            idx = [*range(train_frameset.num_frames)]
            random.shuffle(idx)
            idx = idx[:self.test_batch_size]

            frames = [train_frameset[i] for i in idx]

            pbar_inner = tqdm_notebook(frames, desc='testing on frame', leave=False)

            for frame in pbar_inner:
                self.current_iteration += 1
                strDesc = f'Training on frame {frame.strFrameID}'
                pbar.set_description_str(strDesc)

                # Get the frame
                npFrameBuffer = frame.GetNumpyBuffer()
                torchImageBuffer = torch.FloatTensor(npFrameBuffer)
                torchImageBuffer = torchImageBuffer.unsqueeze(0).to(ptu.GetDevice())
                torchImageBuffer = torchImageBuffer[:, :, :, :3]    # bit of a hack tho

                x = torchImageBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                B, *_ = x.shape
                #print(x.shape)

                # critic update
                self.model.discriminator_optimizer.zero_grad()
                d_loss = self.model.discriminator_loss(x)
                d_loss.backward(retain_graph=True)
                self.model.discriminator_optimizer.step()

                # Generator
                if (self.current_iteration % self.n_critic == 0):
                    self.model.generator_optimizer.zero_grad()
                    # g_loss = -self.model.discriminator_loss(x)
                    g_loss = self.model.generator_loss(x)
                    g_loss.backward()
                    # torch.autograd.set_detect_anomaly(True)
                    self.model.generator_optimizer.step()

                # TODO: both discriminator and generator loss
                self.batch_loss_history.append(d_loss.item())
                strDesc = f'D {d_loss.item():.4f} G {g_loss:.4f} iter {self.current_iteration}'
                pbar_inner.set_description(strDesc)

            self.model.generator_scheduler.step()
            self.model.discriminator_scheduler.step()
            avg_epoch_loss = np.mean(self.batch_loss_history)
            training_losses.append(avg_epoch_loss)
            strDesc = f'D loss {avg_epoch_loss:.4f} iter {self.current_iteration}'
            pbar.set_description(strDesc)

            if (epoch % self.checkpoint_epochs == 0):
                self.SaveCheckpoint(self.checkpoint_file_name, epoch)

        training_losses = np.array(training_losses)
        return training_losses


    def train_gan_epochs(self, train_data, fVerbose=False):
        training_losses = []
        validation_losses = []
        self.current_iteration = 0
        g_loss = 0

        train_loader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        for epoch in pbar:
            self.model.generator.train()
            self.model.discriminator.train()
            self.batch_loss_history = []

            pbar_inner = tqdm_notebook(train_loader, desc='Batch', leave=False)

            #for batch, (x, y) in enumerate(tqdm_notebook(train_loader, desc='Batch', leave=False)):
            for batch, x in enumerate(pbar_inner):
                self.current_iteration += 1
                x = x.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                #y = y.to(ptu.GetDevice())
                B, *_ = x.shape

                # critic update
                self.model.discriminator_optimizer.zero_grad()
                d_loss = self.model.discriminator_loss(x)
                d_loss.backward(retain_graph=True)
                self.model.discriminator_optimizer.step()

                # Generator
                if (self.current_iteration % self.n_critic == 0):
                    self.model.generator_optimizer.zero_grad()
                    #g_loss = -self.model.discriminator_loss(x)
                    g_loss = self.model.generator_loss(x)
                    g_loss.backward()
                    #torch.autograd.set_detect_anomaly(True)
                    self.model.generator_optimizer.step()

                # TODO: both discriminator and generator loss
                self.batch_loss_history.append(d_loss.item())
                strDesc = f'D {d_loss.item():.4f} G {g_loss:.4f} iter {self.current_iteration}'
                pbar_inner.set_description(strDesc)


            self.model.generator_scheduler.step()
            self.model.discriminator_scheduler.step()
            avg_epoch_loss = np.mean(self.batch_loss_history)
            training_losses.append(avg_epoch_loss)
            strDesc = f'D loss {avg_epoch_loss:.4f} iter {self.current_iteration}'
            pbar.set_description(strDesc)

            if(epoch % self.checkpoint_epochs == 0):
                self.SaveCheckpoint(self.checkpoint_file_name, epoch)

        training_losses = np.array(training_losses)
        return training_losses

    def visualize_train_test_plot(self, training_losses, test_losses, strTitle="Train and Test Loss Plot"):
        plt.figure()
        n_epochs = len(test_losses) - 1
        x_train = np.linspace(0, n_epochs, len(training_losses))
        x_test = np.arange(n_epochs + 1)

        plt.plot(x_train, training_losses, label='train loss')
        plt.plot(x_test, test_losses, label='test loss')
        plt.legend()
        plt.title(strTitle)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')


