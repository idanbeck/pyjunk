from repos.pyjunk.junktools import utils
#from repos.pyjunk.junktools.image import image

import imageio
import numpy as np
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from enum import Enum


class image_transform():
    def __init__(self, fVerbose = False, *args, **kwargs):
        super(image_transform, self).__init__(*args, **kwargs)
        self.fVerbose = fVerbose

    def process_transform(self, inImage):
        raise NotImplementedError

# Resize Transform
class image_transform_resize(image_transform):
    def __init__(self, width, height, *args, **kwargs):
        super(image_transform_resize, self).__init__(*args, **kwargs)
        self.width = width
        self.height = height

    def process_transform(self, inImage):
        if (self.fVerbose == True):
            print("resizing image")

        inImage.npImageBuffer = resize(
            inImage.npImageBuffer,
            (self.width, self.height)
        )

# Square Transform
class image_transform_square(image_transform):
    def __init__(self, max_size=None, *args, **kwargs):
        super(image_transform_square, self).__init__(*args, **kwargs)
        self.max_size = max_size

    def process_transform(self, inImage):
        if (self.fVerbose == True):
            print("squaring image state: %s channel: %s id: %s frameset: %s" % (inImage.load_state, inImage.strChannelName, inImage.strFrameID, inImage.strFramesetName))

        width, height, channels = inImage.shape()
        dim = self.get_dim(width, height, self.max_size)

        inImage.npImageBuffer = resize(
            inImage.npImageBuffer,
            (dim, dim)
        )

    def get_dim(self, width, height, max_size=None):
        dim = min(width, height)

        if (max_size != None):
            dim = min(max_size, dim)

        return dim

# Whiten Transform
class image_transform_whiten(image_transform):
    def __init__(self, fZCA, *args, **kwargs):
        super(image_transform_whiten, self).__init__(*args, **kwargs)
        self.fZCA = fZCA

    def process_transform(self, inImage):
        if (self.fVerbose == True):
            print("whitening image state: %s channel: %s id: %s frameset: %s" % (inImage.load_state, inImage.strChannelName, inImage.strFrameID, inImage.strFramesetName))

        width, height, channels = inImage.shape()

        # First get into correct set up (w, h last dims)
        X = inImage.npImageBuffer

        if(X.ndim > 2):
            X.transpose(0, 1, 2)

        # Put into a design matrix
        X = X.reshape(1, width * height * channels)
        # print(X.shape)

        # Center and normalize
        u = X.mean()
        std_dev = X.std()

        # If image is entirely black we get NaNaNaNaNaNaBatMan
        if(u > 1e-5 and std_dev > 1e-5):
            X = (X - u) / std_dev

        # Save this for multi-image
        # Global contrast norm (L2) ensures vector sums to 1
        # X = X / np.sqrt((X ** 2).sum(axis=1))[:, None]

        # ZCA Whitening
        if (self.fZCA):
            # idea: average across channels, calc the ZCA matrix and then apply this to the original image
            cov = X.T.dot(X)
            U, S, V = np.linalg.svd(cov)
            eps = 1e-2  # Prevent division by zero

            # Apply ZCA Whitening matrix
            X = U.dot(np.diag(1.0 / np.sqrt(S + eps))).dot(U.T).dot(X.T).T

        # Reshape back into correct image
        X = X.reshape(width, height, channels)

        # Rescale to [0, 1]
        # If entirely black more nananananabatman
        if (u > 1e-5 and std_dev > 1e-5):
            min, Max = X.min(), X.max()
            X = (X - min) / (Max - min)

        inImage.npImageBuffer = X


