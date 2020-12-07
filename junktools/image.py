from repos.pyjunk.junktools import utils

import imageio
import numpy as np
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

class image():
    def __init__(self, strFilename=None, torchBuffer=None):

        if(strFilename != None):
            strFilepath = utils.get_data_dir(strFilename)
            self.npImageBuffer = np.array(imageio.imread(strFilepath))
            width, height, channels = self.shape()
            self.npImageBuffer = resize(self.npImageBuffer, (width, height))

            # Drop the alpha channel if it exists
            if(channels > 3):
                print("removing alpha channel")
                self.npImageBuffer = self.npImageBuffer[:, :, 0:3]

        elif(torchBuffer != None):
            self.npImageBuffer = torchBuffer.detach().cpu().numpy()
        else:
            raise NotImplementedError


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
        vals = (torch.FloatTensor(self.npImageBuffer)).permute(2, 0, 1)
        grid_image = make_grid(vals, nrow=1)
        plt.figure()
        plt.title(strTitle)
        plt.imshow(grid_image.permute(1, 2, 0))
        plt.axis('off')

    def resize(self, width, height):
        self.npImageBuffer = resize(self.npImageBuffer, (width, height))

    def square(self, max_size=None):
        width, height, channels = self.shape()
        dim = min(width, height)

        if(max_size != None):
            dim = min(max_size, dim)

        return self.resize(dim, dim)

    # Normalizes per the mean and std-dev of the image
    # Utilizes zero component analysis
    def whiten(self, fZCA=False):
        width, height, channels = self.shape()

        # First get into correct set up (w, h last dims)
        X = self.npImageBuffer
        X.transpose(0, 1, 2)

        # Put into a design matrix
        X = X.reshape(1, width * height * channels)
        #print(X.shape)

        # Center and normalize
        u = X.mean()
        std_dev = X.std()
        X = (X - u) / std_dev

        # Save this for multi-image
        # Global contrast norm (L2) ensures vector sums to 1
        #X = X / np.sqrt((X ** 2).sum(axis=1))[:, None]

        # ZCA Whitening
        if(fZCA):
            # idea: average across channels, calc the ZCA matrix and then apply this to the original image
            cov = X.T.dot(X)
            U, S, V = np.linalg.svd(cov)
            eps = 1e-2  # Prevent division by zero

            # Apply ZCA Whitening matrix
            X = U.dot(np.diag(1.0/np.sqrt(S + eps))).dot(U.T).dot(X.T).T

        # Reshape back into correct image
        X = X.reshape(width, height, channels)

        # Rescale to [0, 1]
        min, Max = X.min(), X.max()
        X = (X - min) / (Max - min)

        self.npImageBuffer = X

    def print(self, w=None, h=None, c=None):
        print(self.npImageBuffer[w, h, c])