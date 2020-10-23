import torch
import numpy as np

"""
GPU Related utilities
"""

g_fUseGPU = False
g_torchDevice = None
g_torchGPUID = 0

def SetTorchDevice(gpu_id):
    torch.cuda.set_device(gpu_id)

def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.FloatTensor(*args, **kwargs).to(torch_device)

def FloatTensorFromNumpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(g_torchDevice)

def GetNumpyFromTorchTensor(tTorch):
    return tTorch.to('cpu').detach().numpy()

