import os
from os.path import join, dirname, exists
import numpy as np

from repos.pyjunk.junktools import utils
from repos.pyjunk.junktools.frame import frame

class frameset():
    def __init__(self, strFramesetName=None, num_frames=None, *args, **kwargs):
        super(frameset, self).__init__(*args, **kwargs)
        self.frames = []
        self.num_frames = num_frames
        self.strFramesetName = strFramesetName

        self.LoadFrames(strFramesetName)

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

        print("Loading frames %d to %d" % (self.start_frame, self.end_frame))

        for frame_count in range(self.start_frame, self.end_frame):
            print("Loading: frame %d" % frame_count)
            strFrameID = str(frame_count)
            tempFrame = frame(strFramesetName=self.strFramesetName, strFrameID=strFrameID)
            self.frames.append(tempFrame)

    def Print(self):
        print("Frameset: %s , %d frames" % (self.strFramesetName, len(self.frames)))

    def square(self, max_size):
        for frame in self.frames:
            print("Squaring frame %s" % frame.frame_id())
            frame.square(max_size=max_size)

    def whiten(self, fZCA=False):
        for frame in self.frames:
            print("Whitening frame %s" % frame.frame_id())
            frame.whiten(fZCA=fZCA)

    def visualize(self):
        for frame in self.frames:
            print("Visualizing frame %s" % frame.frame_id())
            frame.visualize()