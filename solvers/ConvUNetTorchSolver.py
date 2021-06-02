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
from repos.pyjunk.junktools.image import image

# ConvVAE Solver class

class ConvUNetTorchSolver(TorchSolver):
    def __init__(self, model, params, fEnableScheduler=False, *args, **kwargs):
        super(ConvUNetTorchSolver, self).__init__(model=model, params=params, *args, *kwargs)
        self.fEnableScheduler = fEnableScheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                           lambda epoch: (self.epochs - epoch) / self.epochs,
                                                           last_epoch=-1)

    def train_frameset(self, train_frameset, train_target_frameset):
        self.model.train()

        training_losses = []

        idx = [*range(train_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.batch_size]
        #print("training on frames %s in frameset %s" % (idx, train_frameset.strFramesetName))

        frames = [train_frameset[i] for i in idx]
        target_frames = [train_target_frameset[i] for i in idx]

        # pbar = tqdm_notebook(zip(frames, target_frames), desc='training on frame', leave=False, total=len(frames))
        # for frame, cond_frame in pbar:
        #     strDesc = f'training on frame {frame.strFrameID}'
        #     pbar.set_description(strDesc)
        #     try:
        #         loss = self.model.loss_with_frame(frame, cond_frame)
        #     except Exception as e:
        #         print(f'failed to load frame {frame.strFrameID}, skipping...')
        #         continue
        #
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #
        #     if(self.grad_clip):
        #         nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
        #
        #     self.optimizer.step()
        #     training_losses.append(loss.item())
        #loss = self.model.loss_with_frames(frames, target_frames)
        try:
            loss = self.model.loss_with_frames(frames, target_frames)
        except Exception as e:
            print(f'epoch failed: {e}, skipping...')
            return [0]

        self.optimizer.zero_grad()
        loss.backward()

        if(self.grad_clip):
            nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        
        if(self.fEnableScheduler):
            self.scheduler.step()
        
        training_losses.append(loss.item())

        return training_losses

    def test_frameset(self, test_frameset, test_target_frameset):
        self.model.eval()
        loss = 0.0

        idx = [*range(test_frameset.num_frames)]
        random.shuffle(idx)
        idx = idx[:self.test_batch_size]
        #print("testing on frames %s in frameset %s" % (idx, test_frameset.strFramesetName))

        frames = [test_frameset[i] for i in idx]
        target_frames = [test_target_frameset[i] for i in idx]

        with torch.no_grad():
            #loss += self.model.loss_with_frameset_and_target(test_source_frameset, test_target_frameset)

            pbar = tqdm_notebook(zip(frames, target_frames), desc='testing on frame', leave=False, total=len(frames))
            for frame, cond_frame in pbar:
                strDesc = f'testing on frame {frame.strFrameID}'
                pbar.set_description(strDesc)
                try:
                    loss += self.model.loss_with_frame(frame, cond_frame)
                except Exception as e:
                    print(f'failed to load frame {frame.strFrameID}: {e}, skipping')
                    continue

            loss /= self.test_batch_size

        #loss /= len(test_data)

        testImgSource = None
        testImgTarget = None
        if(self.save_test_file_name != None):
            frameid = random.randint(0, len(frames) - 1)

            npFrameBuffer = frames[frameid].GetNumpyBuffer()
            torchImageBuffer = torch.FloatTensor(npFrameBuffer)
            torchImageBuffer = torchImageBuffer.squeeze()
            testImgSource = image(torchBuffer=torchImageBuffer)

            testImgTarget = self.model.forward_with_frame(frames[frameid])
        if(loss == 0.0):
            return 0.0, testImgSource, testImgTarget
        else:
            return loss.item(), testImgSource, testImgTarget

    def train_for_epochs_frameset(self,
                                  train_frameset, train_target_frameset,
                                  test_frameset, test_target_frameset,
                                  fVerbose=False):
        training_losses = []
        test_losses = []

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)

        for epoch in pbar:
            train_losses = self.train_frameset(
                train_frameset=train_frameset, train_target_frameset=train_target_frameset
            )
            training_losses.extend(train_losses)

            test_loss, testImgSource, testImgTarget = self.test_frameset(test_frameset, test_target_frameset)
            test_losses.append(test_loss)

            if(fVerbose):
                cur_lr = self.optimizer.param_groups[0]['lr']
                strDesc = f'Epoch {epoch}, Test loss {test_loss:.4f} lr: {cur_lr}'
                pbar.set_description(strDesc)

            if(self.checkpoint_file_name != None and epoch % self.checkpoint_epochs == 0 and epoch != 0):
                print("Saving checkpoint to %s at epoch %s and loss %s" % (self.checkpoint_file_name, epoch, test_loss))

                self.SaveCheckpoint(
                    self.checkpoint_file_name,
                    epoch=epoch,
                    loss=test_loss
                )

                if(self.save_test_file_name != None and testImgSource != None and testImgTarget != None):
                    strSaveFileNameSource = self.save_test_file_name + '_src.png'
                    strSaveFileNameTarget = self.save_test_file_name + '_target.png'

                    testImgSource.SaveToFile(self.strSaveFileNameTarget)
                    testImgTarget.SaveToFile(self.strSaveFileNameTarget)


        return training_losses, test_losses

    def train_frameset_and_plot_losses(self,
                       train_frameset, train_target_frameset,
                       test_frameset, test_target_frameset,
                       strTitle="Train and Test Loss Plot",
                       fVerbose=False):
        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs_frameset(
            train_frameset=train_frameset, train_target_frameset=train_target_frameset,
            test_frameset=test_frameset, test_target_frameset=test_target_frameset,
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

