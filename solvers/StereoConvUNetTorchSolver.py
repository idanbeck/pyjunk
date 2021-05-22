import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import numpy as np

from repos.pyjunk.junktools import utils
from tqdm import trange, tqdm_notebook

from repos.pyjunk.solvers.TorchSolver import TorchSolver

# Stereo ConvVAE Solver class

class StereoConvUNetTorchSolver(TorchSolver):
    def __init__(self, model, params, *args, **kwargs):
        super(StereoConvUNetTorchSolver, self).__init__(model=model, params=params, *args, *kwargs)

    def train_frameset(self, train_left_frameset, train_right_frameset, train_target_frameset):
        self.model.train()

        training_losses = []

        idx = [*range(train_left_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.batch_size]
        #print("training on frames %s in frameset %s" % (idx, train_frameset.strFramesetName))

        frames_left = [train_left_frameset[i] for i in idx]
        frames_right = [train_right_frameset[i] for i in idx]
        target_frames = [train_target_frameset[i] for i in idx]

        pbar = tqdm_notebook(zip(frames_left, frames_right, target_frames), desc='training on frame', leave=False, total=len(frames_left))

        for frame_left, frame_right, target_frame in pbar:
            strDesc = f'training on frame {frame_left.strFrameID}'
            pbar.set_description(strDesc)
            loss = self.model.loss_with_frame(frame_left, frame_right, target_frame)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.grad_clip):
                nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            training_losses.append(loss.item())

        return training_losses

    def test_frameset(self, test_left_frameset, test_right_frameset, test_target_frameset):
        self.model.eval()
        loss = 0.0

        idx = [*range(test_left_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.test_batch_size]
        #print("testing on frames %s in frameset %s" % (idx, test_frameset.strFramesetName))

        frames_left = [test_left_frameset[i] for i in idx]
        frames_right = [test_right_frameset[i] for i in idx]
        target_frames = [test_target_frameset[i] for i in idx]

        with torch.no_grad():
            #loss += self.model.loss_with_frameset_and_target(test_source_frameset, test_target_frameset)

            pbar = tqdm_notebook(zip(frames_left, frames_right, target_frames), desc='testing on frame', leave=False, total=len(frames_left))
            for frame_left, frame_right, target_frame in pbar:
                strDesc = f'testing on frame {frame_left.strFrameID}'
                pbar.set_description(strDesc)
                loss += self.model.loss_with_frame(frame_left, frame_right, target_frame)

            loss /= self.test_batch_size

        #loss /= len(test_data)

        testImg = None
        if(self.save_test_file_name != None):
            frameid = random.randint(0, len(frames_left) - 1)
            testImg = self.model.forward_with_frame(frames_left[frameid], frames_right[frameid])

        return loss.item(), testImg

    def train_for_epochs_frameset(self,
                                  train_left_frameset, train_right_frameset, train_target_frameset,
                                  test_left_frameset, test_right_frameset, test_target_frameset,
                                  fVerbose=False):
        training_losses = []
        test_losses = []

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        for epoch in pbar:
            train_losses = self.train_frameset(
                train_left_frameset=train_left_frameset,
                train_right_frameset=train_right_frameset,
                train_target_frameset=train_target_frameset
            )
            training_losses.extend(train_losses)

            test_loss, testImg = self.test_frameset(
                test_left_frameset=test_left_frameset,
                test_right_frameset=test_right_frameset,
                test_target_frameset=test_target_frameset
            )
            test_losses.append(test_loss)

            if(fVerbose):
                strDesc = f'Epoch {epoch}, Test loss {test_loss:.4f}'
                pbar.set_description(strDesc)

            if(self.checkpoint_file_name != None and epoch % self.checkpoint_epochs == 0 and epoch != 0):
                print("Saving checkpoint to %s at epoch %s and loss %s" % (self.checkpoint_file_name, epoch, test_loss))

                self.SaveCheckpoint(
                    self.checkpoint_file_name,
                    epoch=epoch,
                    loss=test_loss
                )

                if(self.save_test_file_name != None and testImg != None):
                    testImg.SaveToFile(self.save_test_file_name)


        return training_losses, test_losses

    def train_frameset_and_plot_losses(self,
                       train_left_frameset, train_right_frameset, train_target_frameset,
                       test_left_frameset, test_right_frameset, test_target_frameset,
                       strTitle="Train and Test Loss Plot",
                       fVerbose=False):
        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs_frameset(
            train_left_frameset=train_left_frameset,
            train_right_frameset=train_right_frameset,
            train_target_frameset=train_target_frameset,
            test_left_frameset=test_left_frameset,
            test_right_frameset=test_right_frameset,
            test_target_frameset=test_target_frameset,
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

