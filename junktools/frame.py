import os
from os.path import join, dirname, exists
import numpy as np

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.image import image

class frame():
    def __init__(self, strFrameID=None, strFramesetName=None, sourceFrame=None, sourceChannels=None, *args, **kwargs):
        super(frame, self).__init__(*args, **kwargs)

        self.channels = {}
        self.strFrameID = ""
        self.strFramesetName = ""

        # Load from from folder on disk
        if(strFrameID != None):
            self.strFrameID = strFrameID
            self.strFramesetName = strFramesetName
            self.LoadFrame()
        elif(sourceFrame != None):
            if(type(sourceFrame) is not frame):
                raise Exception("Source frame not valid frame object")

            self.strFrameID = sourceFrame.strFrameID
            self.strFramesetName = sourceFrame.strFramesetName

            # Cherry pick the intended channels
            if(sourceChannels != None):
                for strChannel in sourceChannels:
                    self.channels[strChannel] = sourceFrame.channels[strChannel]
            else:
                self.channels = sourceFrame.channels

        else:
            raise NotImplementedError

    def frame_id(self):
        return self.strFrameID

    def __getitem__(self, key):
        return self.channels[key]

    def LoadFrame(self):
        # Enumerate files in the respective folder location
        # pyjunk/frames/

        files, strPath = utils.enum_frame_dir(strFramesetName=self.strFramesetName, strFrameID=self.strFrameID)

        for strFilename in files:
            strName = os.path.splitext(strFilename)[0]
            self.channels[strName] = image(strFilepath=join(strPath, strFilename))

    def visualize(self, strTitle=None):
        for strName, channelImage in self.channels.items():
            if(self.strFrameID != None):
                strName = strName + ": " + self.strFrameID

            if (strTitle != None):
                channelImage.visualize(strTitle=strTitle + ': ' + strName)
            else:
                channelImage.visualize(strTitle=strName)

    def square(self, max_size=256):
        for strName, channelImage in self.channels.items():
            channelImage.square(max_size=max_size)

    def whiten(self, fZCA=False):
        for strName, channelImage in self.channels.items():
            channelImage.whiten(fZCA=fZCA)

    def shape(self, strName=None):
        height, width, channels = 0, 0, 0
        if(strName != None):
            return self.channels[strName].shape()
        else:
            for strChannelName, channelImage in self.channels.items():
                H, W, C = channelImage.shape()

                if(height == 0):
                    height = H
                elif(H != height):
                    raise Exception("Height %d of %s doesn't match height %d of frame" % (H, strChannelName, height))

                if (width == 0):
                    width = W
                elif (W != width):
                    raise Exception("Width %d of %s doesn't match width %d of frame" % (W, strChannelName, width))

                channels += C

            return (height, width, channels)

    def GetNumpyBuffer(self, channels=None):

        # Use selective channels
        if(channels != None):
            fFirst = True
            npBuffer = None

            for strName in channels:
                if(fFirst):
                    npBuffer = self.channels[strName].npImageBuffer
                    fFirst = False
                else:
                    npBuffer = np.concatenate((npBuffer, self.channels[strName].npImageBuffer), axis=2)

            return npBuffer
        else:
            fFirst = True
            npBuffer = None

            for strName, channelImage in self.channels.items():
                if(fFirst):
                    npBuffer = channelImage.npImageBuffer
                    fFirst = False
                else:
                    npBuffer = np.concatenate((npBuffer, channelImage.npImageBuffer), axis=2)

            return npBuffer