import os
from os.path import join, dirname, exists
import numpy as np

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.image import image

class frame():
    def __init__(self,
                 strFrameID=None,
                 strFramesetName=None,
                 sourceFrame=None,
                 sourceChannels=None,
                 fJITLoading=False,
                 fVerbose=False,
                 *args, **kwargs):
        super(frame, self).__init__(*args, **kwargs)

        self.channels = {}
        self.strFrameID = ""
        self.strFramesetName = ""
        self.fJITLoading = fJITLoading
        self.fVerbose = fVerbose

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
            self.fJITLoading = sourceFrame.fJITLoading

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

    def SaveFrame(self, strPath, strExtension):

        strFramePath = join(strPath, self.strFrameID)

        # Create a directory for the frame
        if not os.path.exists(strFramePath):
            os.makedirs(strFramePath)

        for idx, strChannelName  in enumerate(self.channels):
            self.channels[strChannelName].SaveImage(strFramePath, strExtension)


    def LoadFrame(self):
        # Enumerate files in the respective folder location
        # pyjunk/frames/

        files, strPath = utils.enum_frame_dir(
            strFramesetName=self.strFramesetName,
            strFrameID=self.strFrameID
        )

        # Load each channel
        for strFilename in files:
            strName, strExt = os.path.splitext(strFilename)
            strName = strName.lower()
            strExt = strExt.lower()

            # TODO: not handling meta data style JSON files yet
            if(strExt == ".json"):
                continue

            #print("loading %s%s" % (strName, strExt))

            self.channels[strName] = image(
                strFrameID = self.strFrameID,
                strFramesetName = self.strFramesetName,
                strChannelName=strName,
                strFilepath = join(strPath, strFilename),
                fJITLoading = self.fJITLoading,
                fVerbose = self.fVerbose
            )

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
            if (self.fVerbose == True):
                print("squaring frameset %s frame %s channel %s" % (self.strFramesetName, self.strFrameID, strName))

            channelImage.square(max_size=max_size)

    def whiten(self, fZCA=False):
        for strName, channelImage in self.channels.items():
            if (self.fVerbose == True):
                print("whitening frameset %s frame %s channel %s" % (self.strFramesetName, self.strFrameID, strName))

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

            # Unload image from memory
    def Unload(self):
        if (self.fVerbose):
            print("unloading frame %s" % self.load_state)

        for strName, channelImage in self.channels.items():
            channelImage.UnloadImage()

    def GetNumpyBuffer(self, channels=None):

        # Use selective channels
        if(channels != None):
            fFirst = True
            npBuffer = None

            for strName in channels:
                if(fFirst):
                    npBuffer = self.channels[strName].GetNumpyBuffer()
                    fFirst = False
                    if (self.channels[strName].fJITLoading == True):
                        self.channels[strName].UnloadImage()
                else:
                    npBuffer = np.concatenate((npBuffer, self.channels[strName].GetNumpyBuffer()), axis=2)
                    if (self.channels[strName].fJITLoading == True):
                        self.channels[strName].UnloadImage()

            return npBuffer
        else:
            fFirst = True
            npBuffer = None

            for strName, channelImage in self.channels.items():
                if(fFirst):
                    npBuffer = channelImage.GetNumpyBuffer()
                    fFirst = False
                    if(channelImage.fJITLoading == True):
                        channelImage.UnloadImage()
                else:
                    npBuffer = np.concatenate((npBuffer, channelImage.GetNumpyBuffer()), axis=2)
                    if (channelImage.fJITLoading == True):
                        channelImage.UnloadImage()

            return npBuffer