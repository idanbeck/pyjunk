import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import numpy as np
import random

from repos.pyjunk.junktools import utils

from repos.pyjunk.solvers.TorchSolver import TorchSolver

# ConvVAE Solver class

class GQNTorchSolver(TorchSolver):
    def __init__(self, model, params, *args, **kwargs):
        super(GQNTorchSolver, self).__init__(model=model, params=params, *args, *kwargs)

    def generate_from_frameset(self, gen_frameset):
        self.model.eval()

        context_idx = [*range(gen_frameset.num_frames)]
        random.shuffle(context_idx)
        context_idx = context_idx[:self.batch_size]
        query_idx = random.randint(0, gen_frameset.num_frames - 1)
        print("Generating from frames %s in frameset %s with query frame: %s" % (
        context_idx, gen_frameset.strFramesetName, query_idx))

        in_frames = [gen_frameset[i] for i in context_idx]
        query_frame = gen_frameset[query_idx]

        with torch.no_grad():
            gen_image = self.model.generate_with_frames(in_frames, query_frame)

        return gen_image, query_frame

    def generate_from_frameset_with_view(self, gen_frameset, q_view):
        self.model.eval()

        context_idx = [*range(gen_frameset.num_frames)]
        random.shuffle(context_idx)
        context_idx = context_idx[:self.batch_size]

        print("Generating from frames %s in frameset %s with query view: %s" % (
        context_idx, gen_frameset.strFramesetName, q_view))

        in_frames = [gen_frameset[i] for i in context_idx]

        # TODO: implement this
        torchQueryViewBuffer = q_view.GetViewPitchYawTorchTensor().unsqueeze(0)

        with torch.no_grad():
            gen_image = self.model.generate_with_frames_view(in_frames, torchQueryViewBuffer)

        return gen_image


    def train_frameset(self, train_frameset, sigma):
        self.model.train()

        training_losses = []

        context_idx = [*range(train_frameset.num_frames)]
        random.shuffle(context_idx)
        context_idx = context_idx[:self.batch_size]
        query_idx = random.randint(0, train_frameset.num_frames - 1)
        print("training on frames %s in frameset %s with query frame: %s" % (context_idx, train_frameset.strFramesetName, query_idx))

        in_frames = [train_frameset[i] for i in context_idx]
        query_frame = train_frameset[query_idx]

        elbo_loss, kl_loss = self.model.loss_with_frames(in_frames, query_frame, sigma)

        self.optimizer.zero_grad()

        # Update optimizer state
        #elbo_loss.backward()
        elbo_loss.mean().backward()

        if(self.grad_clip):
            nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        if (self.scheduler != None):
            self.scheduler.step()

        training_losses.append(elbo_loss.item())

        return training_losses

    def test_frameset(self, test_frameset, sigma):
        self.model.eval()
        loss = 0.0

        context_idx = [*range(test_frameset.num_frames)]
        random.shuffle(context_idx)
        context_idx = context_idx[:self.batch_size]
        query_idx = random.randint(0, test_frameset.num_frames - 1)
        print("testing on frames %s in frameset %s with query frame: %s" % (
        context_idx, test_frameset.strFramesetName, query_idx))

        in_frames = [test_frameset[i] for i in context_idx]
        query_frame = test_frameset[query_idx]

        # TODO: Unclear if this is the right way to test
        # I think it would make more sense to infer a query view
        # and return the resulting error

        with torch.no_grad():
            elbo_loss, kl_loss = self.model.loss_with_frames(in_frames, query_frame, sigma)

        return elbo_loss.mean().item()

    def train_for_epochs_frameset(self,
                                  train_frameset,
                                  test_frameset,
                                  fVerbose=False):
        training_losses = []
        test_losses = []


        # Pixel standard-deviation
        sigma_i, sigma_f = 2.0, 0.7
        sigma = sigma_i

        for epoch in range(self.epochs):
            train_losses = self.train_frameset(
                train_frameset=train_frameset,
                sigma = sigma
            )
            training_losses.extend(train_losses)

            test_loss = self.test_frameset(test_frameset, sigma)
            test_losses.append(test_loss)

            if(fVerbose):
                print(f'Epoch {epoch}, Test loss {test_loss:.4f}')

            # Pixel-variance annealing
            sigma = max(sigma_f + (sigma_i - sigma_f) * (1 - epoch / (2e5)), sigma_f)

        return training_losses, test_losses

    def train_frameset_and_plot_losses(self,
                       train_frameset,
                       test_frameset,
                       strTitle="Train and Test Loss Plot",
                       fVerbose=False):

        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs_frameset(
            train_frameset=train_frameset,
            test_frameset=test_frameset,
            fVerbose=fVerbose
        )

        # Visualize Plot
        self.visualize_train_test_plot(
            training_losses,
            test_losses,
            strTitle=strTitle)

        # Play a sound when done
        return utils.beep()

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