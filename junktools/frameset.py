import os
from os.path import join, dirname, exists
import numpy as np

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.frame import frame

from repos.pyjunk.junktools.image_transform import image_transform_square

class frameset():
    def __init__(self,
                 strFramesetName=None,
                 num_frames=None,
                 sourceFrameset=None,
                 sourceChannels=None,
                 strNewFramesetName=None,
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

        if(strFramesetName != None):
            self.num_frames = num_frames
            self.strFramesetName = strFramesetName
            self.LoadFrames(strFramesetName)

        elif(sourceFrameset != None):
            self.num_frames = sourceFrameset.num_frames
            self.strFramesetName = strNewFramesetName if strNewFramesetName is not None else sourceFrameset.strFramesetName
            self.fJITLoading = sourceFrameset.fJITLoading
            self.num_channels = len(sourceChannels)
            self.W = sourceFrameset.W
            self.H = sourceFrameset.H
            self.C = sourceFrameset.C

            for sourceFrame in sourceFrameset.frames:
                newFrame = frame(sourceFrame=sourceFrame, sourceChannels=sourceChannels)
                self.frames.append(newFrame)

        else:
            raise NotImplementedError

    def __getitem__(self, key):
        return self.frames[key]

    def LoadFrames(self, strFramesetName):
        # Load frame set from disk, first find the respective json file
        framesetJSON = utils.LoadFramesetJSON(strFramesetName)

        self.start_frame = int(framesetJSON['start_frame'])
        self.end_frame = framesetJSON['end_frame'] if \
            (self.num_frames is None) or (self.start_frame + self.num_frames) >= framesetJSON['end_frame'] \
            else self.start_frame + self.num_frames
        self.end_frame = int(self.end_frame)
        self.strName = framesetJSON['name']
        self.W, self.H, self.C = framesetJSON['shape']
        self.W, self.H, self.C = int(self.W), int(self.H), int(self.C)
        self.num_channels = len(framesetJSON['channels'])
        self.channel_names = framesetJSON['channels']

        # TODO: This needs to be generalized, handled in the image.py code
        # since we drop the alpha channel
        if (self.C > 3):
            self.C = 3


        print("Loading frames %d to %d" % (self.start_frame, self.end_frame))

        for frame_count in range(self.start_frame, self.end_frame):
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
        num_frames = len(self.frames)
        height, width, channels = self.W, self.H, self.C * self.num_channels

        if(self.fJITLoading == False):
            for f in self.frames:
                H, W, C = f.shape()

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

        return (num_frames, height, width, channels)

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

