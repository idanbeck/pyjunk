import os
from os.path import join, dirname, exists
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import json

def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as file:
        picked_data = pickle.load(file)

    train_data, test_data = picked_data['train'], picked_data['test']

    if include_labels:
        train_labels, test_labels = picked_data['train_labels'], picked_data['test_labels']

    # Handle datasets
    # TODO: Generalize this
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize both MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    elif 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]

    # Return the data my bruv
    if include_labels:
        return train_data, test_data, train_labels, test_labels
    else:
        return train_data, test_data

def quantize(images, n_bits):
    images = np.floor(images / 256. * 2 ** n_bits)
    return images.adtype('uint8')

def enum_frame_dir(strFramesetName=None, strFrameID=None):
    strPath = join('repos', 'pyjunk', 'data', 'frames')
    if(strFramesetName != None):
        strPath = join(strPath, strFramesetName)

    if(strFrameID != None):
        # Enumerate the frame folder if exists
        strPath = join(strPath, strFrameID)

    files = os.listdir(strPath)
    return files, strPath

def get_data_dir(strFilename=None):
    if(strFilename != None):
        if '.pkl' in strFilename:
            return join('repos', 'pyjunk', 'data', 'pickles', strFilename)
        elif '.png' in strFilename or '.png' in strFilename:
            return join('repos', 'pyjunk', 'data', 'images', strFilename)
        else:
            return join('repos', 'pyjunk', 'data', strFilename)
    else:
        return join('repos', 'pyjunk', 'data')

def load_mnist(include_labels=False):
    mnist_file_path = get_data_dir('mnist.pkl')

    return load_pickled_data(mnist_file_path, include_labels)

def LoadFramesetJSON(strFramesetName):
    strFramesetFilename = strFramesetName + '.json'
    strPath = join('repos', 'pyjunk', 'data', 'frames', strFramesetName)

    framesetJSONFile = open(join(strPath, strFramesetFilename))
    jsonFrameset = json.load(framesetJSONFile)
    framesetJSONFile.close()
    return jsonFrameset

def visualize_data(data, indexes=None, size=100, nrow=10, nchannels=3):
    if(indexes == None):
        indexes = np.random.choice(len(data), replace=False, size=(size,))

    images = data[indexes].astype('float32') / nchannels * 255.0

    show_samples(images, nrow=nrow)

def show_samples(samples, fname=None, nrow=10, strTitle='Samples'):
    samples = (torch.FloatTensor(samples) / 255.0).permute(0, 3, 1, 2)
    grid_image = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(strTitle)
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.axis('off')

    # optionally save
    if fname is not None:
        if not exists(dirname(fname)):
            os.makedirs(dirname(fname))
        plt.tight_layout()
        plt.savefig(fname)

    plt.show()

