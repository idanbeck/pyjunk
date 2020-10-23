import os
from os.path import join, dirname, exists
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

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

def get_data_dir(strFilename=None):
    if(strFilename != None):
        return join('pytorch', 'data', strFilename)
    else:
        return join('pytorch', 'data')

def load_mnist(include_labels=False):
    mnist_file_path = get_data_dir('mnist.pkl')

    return load_pickled_data(mnist_file_path, include_labels)