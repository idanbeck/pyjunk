from repos.pyjunk.junktools import utils

import imageio
import numpy as np
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from enum import Enum

import os
from os.path import join, dirname, exists

from repos.pyjunk.junktools.image_transform import image_transform_resize
from repos.pyjunk.junktools.image_transform import image_transform_square
from repos.pyjunk.junktools.image_transform import image_transform_whiten

class image():

        class states(Enum):
            not_loaded = "not_loaded",
            loading = "loading",
            loaded = "loaded",
            loaded_from_buffer = "loaded_from_buffer",
            unloading = "unloading"

        def __init__(self,
                     strFilename=None,
                     strFilepath=None,
                     torchBuffer=None,
                     fJITLoading=False,
                     strFrameID=None,
                     strFramesetName=None,
                     strChannelName=None,
                     fVerbose=False):

            self.fJITLoading = fJITLoading
            self.transforms = []
            self.npImageBuffer = None
            self.load_state = self.states.not_loaded
            self.fVerbose = fVerbose

            self.strFrameID = strFrameID
            self.strFramesetName = strFramesetName
            self.strChannelName = strChannelName

            self.strFilepath = strFilepath
            if(strFilename != None):
                self.strFilepath = utils.get_data_dir(strFilename)

            if(self.strFilepath != None):
                if(self.fJITLoading == False):
                    self.LoadImage()

            elif(torchBuffer != None):
                self.npImageBuffer = torchBuffer.detach().cpu().numpy()
                self.load_state = self.states.loaded_from_buffer
            else:
                raise NotImplementedError

        def SaveImage(self, strFramePath, strExtension):
            strFilename = self.strChannelName + '.' + strExtension
            strFramePathName = join(strFramePath, strFilename)
            #imageWriter = imageio.get_writer(strFramePathName)

            # This should work in the context of a JIT mode as well
            npBuffer = self.GetNumpyBuffer()

            if(self.fVerbose):
                print("saving %s" % strFramePathName)
            #print(npBuffer.shape)
            #print(npBuffer)

            #imageWriter.append_data(npBuffer)

            #imageio.imwrite(strFramePathName, npBuffer.astype(np.uint8))
            imageio.imwrite(strFramePathName, (npBuffer * 255.0).astype(np.uint8))

            #imageWriter.close()


        def LoadImage(self):

            self.load_state = self.states.loading
            if(self.fVerbose):
                print("loading image state: %s channel: %s id: %s frameset: %s path: %s" % (self.load_state, self.strChannelName, self.strFrameID, self.strFramesetName, self.strFilepath))

            if (self.strFilepath == None):
                raise NotImplementedError

            self.npImageBuffer = np.array(imageio.imread(self.strFilepath))
            width, height, channels = self.shape()
            self.npImageBuffer = resize(self.npImageBuffer, (width, height))

            # Drop the alpha channel if it exists
            if (channels > 3):
                # print("removing alpha channel")
                self.npImageBuffer = self.npImageBuffer[:, :, 0:3]

            # Apply image transformations
            if(self.fJITLoading == True):
                for transform in self.transforms:
                    transform.process_transform(inImage=self)

            self.load_state = self.states.loaded

            if (self.fVerbose):
                print("loaded image state: %s channel: %s id: %s frameset: %s" % (self.load_state, self.strChannelName, self.strFrameID, self.strFramesetName))

        # Unload image from memory
        def UnloadImage(self):
            if(isinstance(self.npImageBuffer, np.ndarray)):
                self.load_state = self.states.unloading
                del self.npImageBuffer
                self.npImageBuffer = None

            self.load_state = self.states.not_loaded

            if (self.fVerbose):
                print("unloaded image state: %s channel: %s id: %s frameset: %s" % (self.load_state, self.strChannelName, self.strFrameID, self.strFramesetName))

        def GetNumpyBuffer(self):
            if(self.load_state == self.states.not_loaded):
                self.LoadImage()

            if(not isinstance(self.npImageBuffer, np.ndarray)):
                raise BufferError

            return self.npImageBuffer

        def shape(self):
            return self.npImageBuffer.shape

        def width(self):
            return self.npImageBuffer.shape[0]

        def height(self):
            return self.npImageBuffer.shape[1]

        def channels(self):
            return self.npImageBuffer.shape[2]

        def visualize(self, strTitle=None):
            #vals = (torch.FloatTensor(self.npImageBuffer) / 255.0).permute(0, 3, 1, 2)
            #vals = (torch.FloatTensor(self.npImageBuffer) / 255.0).permute(2, 0, 1)

            if (self.fVerbose):
                print("visualizing image state: %s id: %s frameset: %s" % (self.load_state, self.strFrameID, self.strFramesetName))

            npImageBuffer = self.GetNumpyBuffer()
            if(not isinstance(self.npImageBuffer, np.ndarray)):
                raise BufferError

            vals = (torch.FloatTensor(npImageBuffer)).permute(2, 0, 1)
            grid_image = make_grid(vals, nrow=1)
            plt.figure()
            plt.title(strTitle)
            plt.imshow(grid_image.permute(1, 2, 0))
            plt.axis('off')

            if(self.fJITLoading == True):
                self.UnloadImage()

        # Both resize and square are simply a resize transformation
        def resize(self, width, height):
            resize_transform = image_transform_resize(width=width, height=height, fVerbose=self.fVerbose)

            # This will gate the actual implementation to either when JIT is disabled or
            # when it is enabled and the image is currently being loaded
            if(self.fJITLoading == False or self.load_state == self.states.loading):
                resize_transform.process_transform(inImage=self)
            else:
                self.transforms.append(resize_transform)

        def square(self, max_size=None):
            square_transform = image_transform_square(max_size=max_size, fVerbose=self.fVerbose)

            if (self.fJITLoading == False or self.load_state == self.states.loading):
                square_transform.process_transform(inImage=self)
            else:
                self.transforms.append(square_transform)

        # Normalizes per the mean and std-dev of the image
        # Utilizes zero component analysis
        def whiten(self, fZCA=False):
            whiten_transform = image_transform_whiten(fZCA=fZCA, fVerbose=self.fVerbose)

            if (self.fJITLoading == False or self.load_state == self.states.loading):
                whiten_transform.process_transform(inImage=self)
            else:
                self.transforms.append(whiten_transform)


        def print(self, w=None, h=None, c=None):
            print(self.npImageBuffer[w, h, c])