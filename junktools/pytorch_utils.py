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

def HalfTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.HalfTensor(*args, **kwargs).to(torch_device)

def GetFloatTensorFromNumpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(g_torchDevice)

def GetHalfTensorFromNumpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).half().to(g_torchDevice)

def GetNumpyFromTorchTensor(tTorch):
    return tTorch.to('cpu').detach().numpy()


def SetGPUMode(mode, gpu_id=0):
    global g_fUseGPU
    global g_torchDevice
    global g_torchGPUID
    g_torchGPUID = gpu_id
    g_fUseGPU = mode
    g_torchDevice = torch.device("cuda:" + str(g_torchGPUID) if g_fUseGPU else "cpu")

def IsGPUEnabled():
    return g_fUseGPU

def SetDevice(gpu_id):
    torch.cuda.set_device(gpu_id)

def GetDevice():
    return g_torchDevice

def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.randn(*args, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = g_torchDevice
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(g_torchDevice)

