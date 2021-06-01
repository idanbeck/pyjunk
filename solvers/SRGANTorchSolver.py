import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torch.utils.data as data
from repos.pyjunk.junktools.image import image

import numpy as np
import random

from tqdm import trange, tqdm_notebook

from repos.pyjunk.junktools import utils
import repos.pyjunk.junktools.pytorch_utils  as ptu

from repos.pyjunk.solvers.SGANTorchSolver import SGANTorchSolver


class SRGANTorchSolver(SGANTorchSolver):
    def __init__(self, *args, **kwargs):
        super(SRGANTorchSolver, self).__init__(*args, **kwargs)

    # This doesn't train the GAN (only the ResNets etc)
    def train_res_epochs_frameset(self, frameset_lr, frameset_hr, fVerbose=False):
        training_losses = []
        self.current_iteration = 0
        g_loss = 0
        strDesc = ""
        self.model.generator.train()
        optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr)

        pbar = tqdm_notebook(range(int(self.epochs)), desc='Epoch', leave=False)
        for epoch in pbar:
            self.batch_loss_history = []

            idx = [*range(frameset_lr.num_frames)]
            random.shuffle(idx)
            idx = idx[:self.batch_size]

            # Will get the same frames for lr and hr
            frames_lr = [frameset_lr[i] for i in idx]
            frames_hr = [frameset_hr[i] for i in idx]

            pbar_inner = tqdm_notebook(zip(frames_lr, frames_hr), desc='testing on frame', leave=False,
                                       total=len(frames_lr))

            x_lr = None
            x_hr = None
            fFail = False

            for frame_lr, frame_hr in pbar_inner:
                self.current_iteration += 1
                strDescInner = f'Loading frame {frame_lr.strFrameID}'
                pbar_inner.set_description_str(strDescInner)

                # Get the frame
                while (True):
                    try:
                        npFrameLRBuffer = frame_lr.GetNumpyBuffer()
                        break
                    except Exception as e:
                        print(f'failed to load frame {frame_lr.strFrameID}: {e}, skipping')
                        newIdx = random.randint(0, frameset_lr.num_frames)
                        frame_lr = frameset_lr[newIdx]
                        frame_hr = frameset_hr[newIdx]

                torchImageLRBuffer = torch.FloatTensor(npFrameLRBuffer)
                torchImageLRBuffer = torchImageLRBuffer.unsqueeze(0).to(ptu.GetDevice())
                torchImageLRBuffer = torchImageLRBuffer[:, :, :, :3]  # bit of a hack tho
                torchImageLRBuffer = torchImageLRBuffer.permute(0, 3, 1, 2)

                x_lr_ = torchImageLRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0

                if (x_lr == None):
                    x_lr = x_lr_
                else:
                    x_lr = torch.cat((x_lr, x_lr_), dim=0)
                # print(x_lr.shape)

                # Get the low res frame
                try:
                    npFrameHRBuffer = frame_hr.GetNumpyBuffer()
                except Exception as e:
                    print(f'failed to load hr frame {frame_hr.strFrameID}: {e}, skipping')
                    fFail = True
                    break
                torchImageHRBuffer = torch.FloatTensor(npFrameHRBuffer)
                torchImageHRBuffer = torchImageHRBuffer.unsqueeze(0).to(ptu.GetDevice())
                torchImageHRBuffer = torchImageHRBuffer[:, :, :, :3]  # bit of a hack tho
                torchImageHRBuffer = torchImageHRBuffer.permute(0, 3, 1, 2)

                x_hr_ = torchImageHRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                if (x_hr == None):
                    x_hr = x_hr_
                else:
                    x_hr = torch.cat((x_hr, x_hr_), dim=0)
                # print(x_hr.shape)

            if (fFail == True):
                print("skipping epoch, irrecoverable error in image loading")
                continue

            # print(x_lr.shape)
            # print(x_hr.shape)

            # Generator

            pbar.set_description_str('Gen Update:' + strDesc)
            optimizer.zero_grad()
            hr_fake = self.model.generator.forward(x_lr)
            g_loss = F.mse_loss(x_hr, hr_fake)
            g_loss.backward()
            # torch.autograd.set_detect_anomaly(True)
            optimizer.step()

            # # TODO: both discriminator and generator loss
            self.batch_loss_history.append(g_loss.item())
            strDesc = f'G {g_loss.item():.4f} iter: {self.current_iteration}'
            pbar_inner.set_description(strDesc)

            avg_epoch_loss = np.mean(self.batch_loss_history)
            training_losses.append(avg_epoch_loss)
            strDesc = f'G {avg_epoch_loss:.4f} iter {self.current_iteration}'
            pbar.set_description(strDesc)

            if(self.checkpoint_file_name != None and epoch % self.checkpoint_epochs == 0 and epoch != 0):
                print("Saving checkpoint to %s at epoch %s and loss %s" % (self.checkpoint_file_name, epoch, avg_epoch_loss))

                self.SaveCheckpoint(self.checkpoint_file_name, epoch)

                if (self.save_test_file_name != None and hr_fake[0] != None):
                    # torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
                    torchOutput = hr_fake[0].squeeze().permute(1, 2, 0) * 0.5 + 0.5
                    image(torchBuffer=torchOutput).SaveToFile(self.save_test_file_name)

    def train_gan_epochs_frameset(self, frameset_lr, frameset_hr, fVerbose=False):
        training_losses = []
        self.current_iteration = 0
        g_loss = 0
        strDesc = ""

        pbar = tqdm_notebook(range(self.epochs), desc='Epoch', leave=False)
        for epoch in pbar:
            self.model.generator.train()
            self.model.discriminator.train()
            self.batch_loss_history = []

            idx = [*range(frameset_lr.num_frames)]
            random.shuffle(idx)
            idx = idx[:self.batch_size]

            # Will get the same frames for lr and hr
            frames_lr = [frameset_lr[i] for i in idx]
            frames_hr = [frameset_hr[i] for i in idx]

            pbar_inner = tqdm_notebook(zip(frames_lr, frames_hr), desc='testing on frame', leave=False,
                                       total=len(frames_lr))
            #
            # for frame_lr, frame_hr in pbar_inner:
            #     self.current_iteration += 1
            #     strDesc = f'Training on frame {frame_lr.strFrameID}'
            #     pbar.set_description_str(strDesc)
            #
            #     # Get the low res frame
            #     npFrameLRBuffer = frame_lr.GetNumpyBuffer()
            #     torchImageLRBuffer = torch.FloatTensor(npFrameLRBuffer)
            #     torchImageLRBuffer = torchImageLRBuffer.unsqueeze(0).to(ptu.GetDevice())
            #     torchImageLRBuffer = torchImageLRBuffer[:, :, :, :3]    # bit of a hack tho
            #     torchImageLRBuffer = torchImageLRBuffer.permute(0, 3, 1, 2)
            #
            #     x_lr = torchImageLRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
            #     B, *_ = x_lr.shape
            #     #print(x_lr.shape)
            #
            #     # Get the low res frame
            #     npFrameHRBuffer = frame_hr.GetNumpyBuffer()
            #     torchImageHRBuffer = torch.FloatTensor(npFrameHRBuffer)
            #     torchImageHRBuffer = torchImageHRBuffer.unsqueeze(0).to(ptu.GetDevice())
            #     torchImageHRBuffer = torchImageHRBuffer[:, :, :, :3]  # bit of a hack tho
            #     torchImageHRBuffer = torchImageHRBuffer.permute(0, 3, 1, 2)
            #
            #     x_hr = torchImageHRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
            #     # print(x_hr.shape)

            x_lr = None
            x_hr = None
            fFail = False

            for frame_lr, frame_hr in pbar_inner:
                self.current_iteration += 1
                strDescInner = f'Loading frame {frame_lr.strFrameID}'
                pbar_inner.set_description_str(strDescInner)

                # Get the low res frame
                # try:
                #     npFrameLRBuffer = frame_lr.GetNumpyBuffer()
                # except Exception as e:
                #     print(f'failed to load lr frame {frame_lr.strFrameID}, skipping')
                #     continue

                # Get the frame
                while (True):
                    try:
                        npFrameLRBuffer = frame_lr.GetNumpyBuffer()
                        break
                    except Exception as e:
                        print(f'failed to load frame {frame_lr.strFrameID}: {e}, skipping')
                        newIdx = random.randint(0, frameset_lr.num_frames)
                        frame_lr = frameset_lr[newIdx]
                        frame_hr = frameset_hr[newIdx]

                torchImageLRBuffer = torch.FloatTensor(npFrameLRBuffer)
                torchImageLRBuffer = torchImageLRBuffer.unsqueeze(0).to(ptu.GetDevice())
                torchImageLRBuffer = torchImageLRBuffer[:, :, :, :3]  # bit of a hack tho
                torchImageLRBuffer = torchImageLRBuffer.permute(0, 3, 1, 2)

                x_lr_ = torchImageLRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0

                if (x_lr == None):
                    x_lr = x_lr_
                else:
                    x_lr = torch.cat((x_lr, x_lr_), dim=0)
                # print(x_lr.shape)

                # Get the low res frame
                try:
                    npFrameHRBuffer = frame_hr.GetNumpyBuffer()
                except Exception as e:
                    print(f'failed to load hr frame {frame_hr.strFrameID}: {e}, skipping')
                    fFail = True
                    break
                torchImageHRBuffer = torch.FloatTensor(npFrameHRBuffer)
                torchImageHRBuffer = torchImageHRBuffer.unsqueeze(0).to(ptu.GetDevice())
                torchImageHRBuffer = torchImageHRBuffer[:, :, :, :3]  # bit of a hack tho
                torchImageHRBuffer = torchImageHRBuffer.permute(0, 3, 1, 2)

                x_hr_ = torchImageHRBuffer.to(ptu.GetDevice()).float().contiguous() * 2.0 - 1.0
                if (x_hr == None):
                    x_hr = x_hr_
                else:
                    x_hr = torch.cat((x_hr, x_hr_), dim=0)
                # print(x_hr.shape)

            if (fFail == True):
                print("skipping epoch, irrecoverable error in image loading")
                continue

            # print(x_lr.shape)
            # print(x_hr.shape)

            # critic update
            pbar.set_description_str('Critic Update:' + strDesc)
            self.model.discriminator_optimizer.zero_grad()
            d_loss = self.model.discriminator_loss(x_hr, x_lr)
            d_loss.backward(retain_graph=True)
            self.model.discriminator_optimizer.step()

            # Generator

            if (self.current_iteration % self.n_critic == 0):
                pbar.set_description_str('Gen Update:' + strDesc)
                self.model.generator_optimizer.zero_grad()
                # g_loss = -self.model.discriminator_loss(x)
                g_loss, hr_fake = self.model.generator_loss(x_hr, x_lr)
                g_loss.backward()
                # torch.autograd.set_detect_anomaly(True)
                self.model.generator_optimizer.step()

            # # TODO: both discriminator and generator loss
            self.batch_loss_history.append(d_loss.item())
            strDesc = f'D {d_loss.item():.4f} G {g_loss:.4f} iter {self.current_iteration}'
            pbar_inner.set_description(strDesc)

            self.model.generator_scheduler.step()
            self.model.discriminator_scheduler.step()
            avg_epoch_loss = np.mean(self.batch_loss_history)
            training_losses.append(avg_epoch_loss)
            strDesc = f'D {avg_epoch_loss:.4f} G {g_loss:.4f} iter {self.current_iteration}'
            pbar.set_description(strDesc)

            #if (epoch % self.checkpoint_epochs == 0):
            #     self.SaveCheckpoint(self.checkpoint_file_name, epoch)
            if (self.checkpoint_file_name != None and epoch % self.checkpoint_epochs == 0 and epoch != 0):
                print("Saving checkpoint to %s at epoch %s and loss %s" % (
                self.checkpoint_file_name, epoch, avg_epoch_loss))

                self.SaveCheckpoint(self.checkpoint_file_name, epoch)

                if (self.save_test_file_name != None and hr_fake[0] != None):
                    # torchOutput = torchOutput.squeeze().permute(1, 2, 0) * 0.5 + 0.5
                    torchOutput = hr_fake[0].squeeze().permute(1, 2, 0) * 0.5 + 0.5
                    image(torchBuffer=torchOutput).SaveToFile(self.save_test_file_name) 

        training_losses = np.array(training_losses)
        return training_losses
