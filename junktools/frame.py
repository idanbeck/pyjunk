import os
from os.path import join, dirname, exists
import numpy as np
import torch

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.image import image
import repos.pyjunk.junktools.pytorch_utils  as ptu

import json
import math

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
        self.meta = {}
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
            self.meta = sourceFrame.meta

            # Cherry pick the intended channels
            if(sourceChannels != None):
                for strChannel in sourceChannels:
                    strChannel = strChannel.lower()

                    if strChannel not in sourceFrame.channels.keys():
                        print("%s channel not present" % strChannel)
                        print(sourceFrame.channels.keys())
                        raise KeyError

                    self.channels[strChannel] = sourceFrame.channels[strChannel]
            else:
                self.channels = sourceFrame.channels

        else:
            raise NotImplementedError

    def frame_id(self):
        return self.strFrameID

    def __getitem__(self, key):
        return self.channels[key]

    def GetFrameMeta(self, strKey):
        return self.meta[strKey]

    def GetFrameCameraView(self):
        frame_camera_meta = self.GetFrameMeta("camera")
        frame_camera_position = frame_camera_meta['position']
        frame_camera_look_at = frame_camera_meta['look_at']
        x, y, z = frame_camera_position['x'], frame_camera_position['y'], frame_camera_position['z']
        l_x, l_y, l_z = frame_camera_look_at['x'], frame_camera_look_at['y'], frame_camera_look_at['z']

        # View direction is the look at minus the position
        vx, vy, vz = l_x - x, l_y - y, l_z - z
        pitch_rad = math.atan2(vy, vz)      # Pitch is about the x axis
        yaw_rad = math.atan2(vz, vx)        # yaw is about the y axis

        # Note: Might want to confirm these values
        #return torch.tensor(
        return ptu.tensor(
            [x, y, z,
             math.cos(yaw_rad), math.sin(yaw_rad),
             math.cos(pitch_rad), math.sin(pitch_rad)]
        )


    def SaveFrame(self, strPath, strExtension):

        strFramePath = join(strPath, self.strFrameID)

        # Create a directory for the frame
        if not os.path.exists(strFramePath):
            os.makedirs(strFramePath)

        # channels
        for idx, strChannelName  in enumerate(self.channels):
            self.channels[strChannelName].SaveImage(strFramePath, strExtension)

        # meta data
        for idx, strMetaName in enumerate(self.meta):
            strFilename = strMetaName + '.json'
            strJsonFilePath = join(strFramePath, strFilename)

            with open(strJsonFilePath, 'w') as frameMetaJSONFile:
                json.dump(self.meta[strMetaName], frameMetaJSONFile, indent=4)
                frameMetaJSONFile.close()


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
                frameJSONFile = open(join(strPath, strFilename))
                self.meta[strName] = json.load(frameJSONFile)
                frameJSONFile.close()
            else:
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

    def clear_transforms(self):
        for strName, channelImage in self.channels.items():
            if (self.fVerbose == True):
                print("clearing frameset %s frame %s channel %s transform" % (self.strFramesetName, self.strFrameID, strName))

            channelImage.clear_transforms()

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