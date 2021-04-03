import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import numpy as np
from tqdm import trange, tqdm_notebook

from repos.pyjunk.junktools import utils

from repos.pyjunk.solvers.TorchSolver import TorchSolver

class SimpleTorchSolver(TorchSolver):
    def __init__(self, model, params, *args, **kwargs):
        super(SimpleTorchSolver, self).__init__(model=model, params=params, *args, *kwargs)

    def train_images(self, train_data_images):
        self.model.train()
        training_losses = []

        for image in train_data_images:
            loss = self.model.loss_with_image(image)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.grad_clip):
                nn.utis.clip_grad_norm(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            training_losses.append(loss.item())

        return training_losses

    def train_frames(self, train_data):
        self.model.train()
        training_losses = []

        sourceFrames, targetFrames = zip(*train_data)
        for sourceFrame, targetFrame in zip(sourceFrames, targetFrames):
            loss = self.model.loss_with_frame_and_target(sourceFrame, targetFrame)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.grad_clip):
                nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            training_losses.append(loss.item())

        return training_losses

    def train_frameset(self, train_source_frameset, train_target_frameset):
        self.model.train()
        training_losses = []

        idx = [*range(train_source_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.batch_size]
        print("training on frames %s in frameset %s" % (idx, train_source_frameset.strFramesetName))

        sourceFrames = [train_source_frameset[i] for i in idx]
        targetFrames = [train_target_frameset[i] for i in idx]

        #print(len(sourceFrames))

        for sourceFrame, targetFrame in zip(sourceFrames, targetFrames):
            #loss = self.model.loss_with_frameset_and_target(train_source_frameset, train_target_frameset)
            loss = self.model.loss_with_frame_and_target(sourceFrame, targetFrame)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.grad_clip):
                nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            training_losses.append(loss.item())

        return training_losses

    def test_images(self, test_data_images):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for image in test_data_images:
                loss += self.model.loss_with_image(image)

            loss /= len(test_data_images)

        return loss.item()

    def test_frames(self, test_data):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            sourceFrames, targetFrames = zip(*test_data)
            for sourceFrame, targetFrame in zip(sourceFrames, targetFrames):
                loss += self.model.loss_with_frame_and_target(sourceFrame, targetFrame)

            loss /= len(test_data)

        return loss.item()

    def test_frameset(self, test_source_frameset, test_target_frameset):
        self.model.eval()
        loss = 0.0

        idx = [*range(test_source_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.test_batch_size]
        print("testing on frames %s in frameset %s" % (idx, test_source_frameset.strFramesetName))

        sourceFrames = [test_source_frameset[i] for i in idx]
        targetFrames = [test_target_frameset[i] for i in idx]

        with torch.no_grad():
            #loss += self.model.loss_with_frameset_and_target(test_source_frameset, test_target_frameset)

            for sourceFrame, targetFrame in zip(sourceFrames, targetFrames):
                loss += self.model.loss_with_frame_and_target(sourceFrame, targetFrame)

            loss /= self.test_batch_size

        #loss /= len(test_data)

        return loss.item()

    # TODO: These two functions (frames/images) are the same formally
    def train_for_epochs_images(self, train_data_images, test_data_images, fVerbose=False):
        training_losses = []
        test_losses = []

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        #for epoch in range(self.epochs):
        for epoch in pbar:
            train_losses = self.train_images(train_data_images)
            training_losses.extend(train_losses)

            test_loss = self.test_images(test_data_images)
            test_losses.append(test_loss)

            if (fVerbose):
                strDesc = f'Epoch {epoch}, Test loss {test_loss:.4f}'
                # print(strDesc)
                pbar.set_description(strDesc)

        pbar.close()

        return training_losses, test_losses

    def train_for_epochs_frames(self, train_data, test_data, fVerbose=False):
        training_losses = []
        test_losses = []

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        #for epoch in range(self.epochs):
        for epoch in pbar:
            train_losses = self.train_frames(train_data)
            training_losses.extend(train_losses)

            test_loss = self.test_frames(test_data)
            test_losses.append(test_loss)

            if(fVerbose):
                strDesc = f'Epoch {epoch}, Test loss {test_loss:.4f}'
                #print(strDesc)
                pbar.set_description(strDesc)

        pbar.close()

        return training_losses, test_losses

    # TODO: utilize tqdm here
    def train_for_epochs_frameset(self,
                                  train_source_frameset, train_target_frameset,
                                  test_source_frameset, test_target_frameset,
                                  fVerbose=False):
        training_losses = []
        test_losses = []

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        #for epoch in range(self.epochs):
        for epoch in pbar:
            train_losses = self.train_frameset(
                train_source_frameset=train_source_frameset,
                train_target_frameset=train_target_frameset,
            )
            training_losses.extend(train_losses)

            test_loss = self.test_frameset(
                test_source_frameset=train_source_frameset,
                test_target_frameset=train_target_frameset,
            )
            test_losses.append(test_loss)

            if(fVerbose):
                strDesc = f'Epoch {epoch}, Test loss {test_loss:.4f}'
                # print(strDesc)
                pbar.set_description(strDesc)

        pbar.close()

        return training_losses, test_losses

    def train_and_visualize_images(self, train_data_images, test_data_images, strTitle="Train, Test Loss Plot", fVerbose=False):
        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs_images(train_data_images, test_data_images, fVerbose=fVerbose)

        # Visualize Plot
        self.visualize_train_test_plot(training_losses, test_losses, strTitle=strTitle)

        # Visualize image
        self.visualize_image_input(test_data_images[0], strTitle=strTitle)

        # Play a sound when done
        return utils.beep()

    def train_and_visualize_frames(self, train_data, test_data, strTitle="Train and Test Loss Plot", fVerbose=False):
        # Train and eval the model
        training_losses, test_losses = self.train_for_epochs_frames(train_data, test_data, fVerbose=fVerbose)

        # Visualize Plot
        self.visualize_train_test_plot(training_losses, test_losses, strTitle=strTitle)

        # Visualize image
        self.visualize_frame_input(test_data[0][0], strTitle=strTitle)

        # Play a sound when done
        return utils.beep()

    def train_and_visualize_frameset(self,
                                     train_source_frameset, train_target_frameset,
                                     test_source_frameset, test_target_frameset,
                                     strTitle="Train and Test Loss Plot",
                                     fVerbose=False):
        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs_frameset(
            train_source_frameset=train_source_frameset,
            train_target_frameset=train_target_frameset,
            test_source_frameset=test_source_frameset,
            test_target_frameset=test_target_frameset,
            fVerbose=fVerbose
        )

        # Visualize Plot
        self.visualize_train_test_plot(training_losses, test_losses, strTitle=strTitle)

        # Visualize Image #TODO: This could be more better
        self.visualize_frameset(test_source_frameset, test_target_frameset, strTitle=strTitle)

        # Play a sound when done
        return utils.beep()

    def visualize_frame_input(self, frameObject, strTitle="frame input"):
        imagePassThru = self.model.forward_with_frame(frameObject)
        imagePassThru.visualize(strTitle=strTitle)

    def visualize_image_input(self, imageObject, strTitle="frame input"):
        imagePassThru = self.model.forward_with_image(imageObject)
        imagePassThru.visualize(strTitle=strTitle)

    def visualize_frameset(self, test_source_frameset, test_target_frameset, strTitle="frameset", idx=0):

        imagePassThru = self.model.forward_with_frame(test_source_frameset[idx])
        test_target_frameset[idx].visualize(strTitle=strTitle + "target")
        imagePassThru.visualize(strTitle=strTitle + "pass thru")

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