import os
from os.path import join, dirname, exists
import numpy as np
import json
import random

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.frame import frame

from repos.pyjunk.junktools.image_transform import image_transform_square
from tqdm import trange, tqdm_notebook

class frameset():
    def __init__(self,
                 strFramesetName=None,
                 num_frames=None,
                 sourceFrameset=None,
                 sourceChannels=None,
                 strNewFramesetName=None,
                 sourceFrames=None,
                 fJITLoading=False,
                 fVerbose=False,
                 *args, **kwargs
                 ):

        super(frameset, self).__init__(*args, **kwargs)
        self.frames = []
        self.num_frames = 0
        self.num_channels = 0
        self.strFramesetName = ""
        self.fJITLoading = fJITLoading
        self.fVerbose = fVerbose
        self.W = 0
        self.H = 0
        self.C = 0
        self.framesetJSON = None
        self._shape = None

        if(strFramesetName != None):
            if(num_frames != None):
                self.num_frames = num_frames
            else:
                self.num_frames = None
            self.strFramesetName = strFramesetName
            self.LoadFrames(strFramesetName)

        elif(sourceFrames != None):
            # This takes a frameset and selects out of it a set of frames
            self.num_frames = num_frames
            self.strFramesetName = strNewFramesetName if strNewFramesetName is not None else sourceFrameset.strFramesetName
            self.fJITLoading = sourceFrameset.fJITLoading
            self.num_channels = sourceFrameset.num_channels
            self.W = sourceFrameset.W
            self.H = sourceFrameset.H
            self.C = sourceFrameset.C

            for sourceFrame in sourceFrames:
                newFrame = frame(sourceFrame=sourceFrame)
                self.frames.append(newFrame)

        elif(sourceFrameset != None):
            # This takes a frameset and reduces it to channels
            self.num_frames = sourceFrameset.num_frames
            self.strFramesetName = strNewFramesetName if strNewFramesetName is not None else sourceFrameset.strFramesetName
            self.fJITLoading = sourceFrameset.fJITLoading
            self.num_channels = len(sourceChannels)
            self.W = sourceFrameset.W
            self.H = sourceFrameset.H
            self.C = sourceFrameset.C
            self.framesetJSON = sourceFrameset.framesetJSON

            for sourceFrame in sourceFrameset.frames:
                newFrame = frame(sourceFrame=sourceFrame, sourceChannels=sourceChannels)
                self.frames.append(newFrame)

        else:
            raise NotImplementedError

    def __getitem__(self, key):
        return self.frames[key]

    # This will create two framesets from this one
    # randomly sampling the frames and splitting per the ratio
    def split_into_train_and_test(self, train_test_ratio=0.8, train_indices=None, test_indices=None):
        if(train_test_ratio > 0.8 and train_test_ratio < 0.2):
            print("ratio cannot be above 80% and must be above 20%")
            raise ValueError

        idx = [*range(self.num_frames)]
        random.shuffle(idx)
        idx_split = int(self.num_frames * train_test_ratio)
        idx_train = idx[:idx_split]
        idx_test = idx[idx_split:]

        # May provide override
        if(train_indices != None):
            idx_train = train_indices

        if(test_indices != None):
            idx_test = test_indices

        trainingFrames = [self.frames[i] for i in idx_train]
        testFrames = [self.frames[i] for i in idx_test]

        strTrainFramesetName = self.strFramesetName + "_train"
        strTestFramesetName = self.strFramesetName + "_test"

        train_frameset = frameset(
            sourceFrames=trainingFrames,
            sourceFrameset=self,
            strNewFramesetName=strTrainFramesetName,
            num_frames=len(idx_train),
            fVerbose=self.fVerbose
        )

        test_frameset = frameset(
            sourceFrames=testFrames,
            sourceFrameset=self,
            strNewFramesetName=strTestFramesetName,
            num_frames=len(idx_test),
            fVerbose=self.fVerbose
        )

        return train_frameset, test_frameset

    def get_frame_ids(self):
        indices = []

        for frame in self.frames:
            indices.append(int(frame.frame_id()) - 1)

        return indices

    def save_to_new_frameset(self, strNewFramesetName, strExtension="png"):
        # Create a new JSON description
        jsonData = {}
        jsonData['name'] = strNewFramesetName
        jsonData['start_frame'] = self.start_frame
        jsonData['end_frame'] = self.end_frame
        jsonData['channels'] = self.channel_names
        jsonData['extension'] = strExtension
        jsonData['shape'] = [self.W, self.H, self.C]

        # Save the JSON file to a new folder named as the respective new frameset
        strJsonPathname, strPath = utils.SaveNewFramesetJSON(strNewFramesetName, jsonData)
        print(strJsonPathname)

        # Save the frames
        print("Saving frames %d to %d" % (self.start_frame, self.end_frame))

        pbar = tqdm_notebook(self.frames, desc="frame", leave=False)

        #for idx, frame in enumerate(self.frames):
        for idx, frame in enumerate(pbar):
            frame.SaveFrame(strPath, strExtension)
            frame.Unload()

        # Play a sound when done
        return utils.beep()


    def LoadFrames(self, strFramesetName):
        # Load frame set from disk, first find the respective json file
        framesetJSON = utils.LoadFramesetJSON(strFramesetName)

        self.start_frame = int(framesetJSON['start_frame'])
        if((self.num_frames is None) or (self.start_frame + self.num_frames) >= framesetJSON['end_frame']):
            self.end_frame = framesetJSON['end_frame']
            self.num_frames = self.end_frame - self.start_frame
        else:
            self.end_frame = self.start_frame + self.num_frames

        self.end_frame = int(self.end_frame)
        self.strName = framesetJSON['name']
        self.W, self.H, self.C = framesetJSON['shape']
        self.W, self.H, self.C = int(self.W), int(self.H), int(self.C)
        self.num_channels = len(framesetJSON['channels'])
        self.channel_names = framesetJSON['channels']

        # For posterity
        self.framesetJSON = framesetJSON

        # TODO: This needs to be generalized, handled in the image.py code
        # since we drop the alpha channel
        if (self.C > 3):
            self.C = 3


        print("Loading frames %d to %d" % (self.start_frame, self.end_frame))

        pbar = tqdm_notebook(range(self.start_frame, self.end_frame), desc="frame", leave=False)

        # for idx, frame in enumerate(self.frames):
        #for frame_count in range(self.start_frame, self.end_frame):
        for frame_count in pbar:
            if (self.fVerbose):
                print("Loading: frame %d" % frame_count)
            strFrameID = str(frame_count)
            tempFrame = frame(strFramesetName=self.strFramesetName,
                              strFrameID=strFrameID,
                              fJITLoading=self.fJITLoading,
                              fVerbose=self.fVerbose)
            self.frames.append(tempFrame)

        # Play a sound when done
        return utils.beep()

    def Print(self):
        #num_frames = self.end_frame - self.start_frame
        print("Frameset: %s, %d frames with %d channels JIT %s" %
              (self.strFramesetName, len(self.frames), self.num_channels, ("enabled" if self.fJITLoading else "disabled")))
        #print("Frameset: %s , %d frames" % (self.strFramesetName, num_frames))

    def clear_transforms(self):
        for frame in self.frames:
            if (self.fVerbose):
                print("clearing frame %s transform" % frame.frame_id())
            frame.clear_transforms()

            # This is a bit hacky
            if(self.framesetJSON != None):
                self.W, self.H, self.C = self.framesetJSON['shape']

    def square(self, max_size):
        for frame in self.frames:
            if (self.fVerbose):
                print("Squaring frame %s" % frame.frame_id())
            frame.square(max_size=max_size)
            # This is a bit hacky
            dim = image_transform_square.get_dim(self.W, self.H, max_size)
            self.W = dim
            self.H = dim

            # Play a sound when done
        return utils.beep()

    def whiten(self, fZCA=False):
        for frame in self.frames:
            if (self.fVerbose):
                print("Whitening frame %s" % frame.frame_id())
            frame.whiten(fZCA=fZCA)

        # Play a sound when done
        return utils.beep()

    def visualize(self, strTitle=None):
        for f in self.frames:

            print("Visualizing frame %s: %s" % (self.strFramesetName, f.frame_id()))

            if (strTitle != None):
                f.visualize(strTitle=strTitle + ': ' + self.strFramesetName)
            else:
                f.visualize(strTitle=self.strFramesetName)

    def shape(self):
        if(self._shape == None):
            num_frames = len(self.frames)
            height, width, channels = self.frames[0].shape()

            if(self.fJITLoading == False):
                for f in self.frames[1:]:
                    H, W, C = f.shape()
                    print(f.shape())

                    if (height == 0):
                        height = H
                    elif (H != height):
                        raise Exception("Height %d of frame %s doesn't match height %d of frameset" % (H, f.strFrameID, height))

                    if (width == 0):
                        width = W
                    elif (W != width):
                        raise Exception("Width %d of frame %s doesn't match width %d of frameset" % (W, f.strFrameID, width))

                    if (channels == 0):
                        channels = C
                    elif (C != channels):
                        raise Exception("Channels %d of frame %s doesn't match channels %d of frameset" % (C, f.strFrameID, channels))

            self._shape = (num_frames, height, width, channels)

        return self._shape

    # TODO: Add select_frames to allow for minibatching
    def GetNumpyBuffer(self, channels=None, select_frames=None):
        fFirst = True
        npBuffer = None

        for f in self.frames:
            npFrameBuffer = f.GetNumpyBuffer(channels=channels)
            if(fFirst):
                npBuffer = npFrameBuffer
                npBuffer = np.expand_dims(npBuffer, axis=0)
                fFirst = False
            else:
                npBuffer = np.concatenate((npBuffer, np.expand_dims(npFrameBuffer, axis=0)), axis=0)

        return npBuffer

