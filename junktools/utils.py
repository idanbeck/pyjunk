import os
from os.path import join, dirname, exists
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import json

import IPython

def beep(freq=440, msDuration=55.0, envelope=None):
    if(envelope != None):
        msAttack, msDecay, ratioSustain, msRelease = envelope

    fSampling = 44100
    npTime = np.linspace(start=0.0,
                         stop=(msDuration/1000.0),
                         num=int((msDuration/1000.0) * fSampling))

    # Release
    if(envelope != None):
        npTime = np.append(
            npTime,
            np.linspace(
                start=(msDuration/1000.0),
                stop=(msDuration/1000.0) + (msRelease / 1000.0),
                num=int((msRelease/1000.0) * fSampling))
        )

    npBeep = np.sin(npTime * freq * (2 * np.pi))

    # Envelope
    if(envelope != None):
        npAttack = np.linspace(start=0.0,
                               stop=1.0,
                               num=int((msAttack/1000.0) * fSampling))
        npDecay = np.linspace(start=1.0,
                              stop=ratioSustain,
                              num=int((msDecay / 1000.0) * fSampling))
        msSustain = msDuration - msDecay - msAttack
        npSustain = np.array([ratioSustain] * int((msSustain / 1000.0) * fSampling))
        npRelease = np.linspace(start=ratioSustain,
                                stop=0.0,
                                num=int((msRelease / 1000.0) * fSampling))
        npEnvelope = np.concatenate((npAttack, npDecay, npSustain, npRelease))

        npBeep = npBeep * npEnvelope

    return IPython.display.Audio(
        npBeep, rate=fSampling, autoplay=True)

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

def load_cifar(include_labels=False):

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Train data
    for n_batch in range(1, 6):
        strFilename = f"data_batch_{n_batch}"
        cifar_file_path = join(
            'repos',
            'pyjunk',
            'data',
            'pickles',
            'cifar-10-batches-py',
            strFilename)

        with open(cifar_file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        train_data.extend(dict[b'data'])
        train_labels.extend(dict[b'labels'])
        fo.close()

    train_data = np.array(train_data)
    train_data = train_data.reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels)

    strFilename = f"test_batch"
    cifar_file_path = join(
        'repos',
        'pyjunk',
        'data',
        'pickles',
        'cifar-10-batches-py',
        strFilename)

    with open(cifar_file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    test_data.extend(dict[b'data'])
    test_labels.extend(dict[b'labels'])
    fo.close()

    test_data = np.array(test_data)
    test_data = test_data.reshape(-1, 3, 32, 32)
    test_labels = np.array(test_labels)

    train_data = train_data.transpose(0, 2, 3, 1) / 255.0
    test_data = test_data.transpose(0, 2, 3, 1) / 255.0

    if(include_labels == False):
        return train_data, test_data
    else:
        return train_data, train_labels, test_data, test_labels

def LoadFramesetJSON(strFramesetName):
    strFramesetFilename = strFramesetName + '.json'
    strPath = join('repos', 'pyjunk', 'data', 'frames', strFramesetName)

    framesetJSONFile = open(join(strPath, strFramesetFilename))
    jsonFrameset = json.load(framesetJSONFile)
    framesetJSONFile.close()
    return jsonFrameset

def SaveNewFramesetJSON(strFramesetName, jsonData):
    strFramesetFilename = strFramesetName + '.json'
    strPath = join('repos', 'pyjunk', 'data', 'frames', strFramesetName)

    if not os.path.exists(strPath):
        os.makedirs(strPath)

    with open(join(strPath, strFramesetFilename), 'w') as framesetJSONFile:
        json.dump(jsonData, framesetJSONFile, indent=4)
        framesetJSONFile.close()

    return join(strPath, strFramesetFilename), strPath

def visualize_data(data, indexes=None, size=100, nrow=10, nchannels=3, fRandom=True):
    if(indexes == None and fRandom == True):
        indexes = np.random.choice(len(data), replace=False, size=(size,))
    else:
        indexes = np.arange(size)

    #images = data[indexes].astype('float32') / nchannels * 255.0
    images = data[indexes].astype('float32') * 255.0
    #images = data[indexes]

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

