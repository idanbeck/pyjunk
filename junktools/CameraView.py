import torch
import math
import numpy as np

# A simple Camera View object

class CameraView():
    def __init__(self, ptPosition=None, vLookAt=(0.0, 0.0, 0.0), radius=None):
        self.ptPosition = ptPosition
        self.vLookAt = vLookAt

        if(radius != None):
            self.ptPosition = np.random.rand(3)
            self.ptPosition = self.ptPosition / np.linalg.norm(self.ptPosition)
            self.ptPosition *= radius
            self.ptPosition = tuple(self.ptPosition)

    def GetViewPitchYawTorchTensor(self):
        x, y, z = self.ptPosition
        l_x, l_y, l_z = self.vLookAt

        # View direction is the look at minus the position
        vx, vy, vz = l_x - x, l_y - y, l_z - z
        pitch_rad = math.atan2(vy, vz)  # Pitch is about the x axis
        yaw_rad = math.atan2(vz, vx)  # yaw is about the y axis

        # Note: Might want to confirm these values
        return torch.tensor(
            [x, y, z,
             math.cos(yaw_rad), math.sin(yaw_rad),
             math.cos(pitch_rad), math.sin(pitch_rad)], dtype=torch.float32
        )

    def Print(self):
        x, y, z = self.ptPosition
        l_x, l_y, l_z = self.vLookAt

        # View direction is the look at minus the position
        vx, vy, vz = l_x - x, l_y - y, l_z - z
        pitch_rad = math.atan2(vy, vz)  # Pitch is about the x axis
        yaw_rad = math.atan2(vz, vx)  # yaw is about the y axis

        print("pt: %s view: %s pitch: %s yaw: %s" %
              (self.ptPosition, self.vLookAt, pitch_rad, yaw_rad))