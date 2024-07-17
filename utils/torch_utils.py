import os
import random
import numpy as np
import torch

from typing import Union, List


AVAILABLE_GPU = torch.cuda.is_available()


def convert_from_tensor(value: torch.Tensor):
    """
    Value to be converted from torch Tensor to python.

    Args:
        value: Tensor to convert

    Returns:
        Numpy array or plain python value.
    """
    try:
        converted_value = value.detach().cpu().item()
    except ValueError:
        converted_value = value.detach().cpu().numpy()
    return converted_value


def get_device(available_gpu: bool = AVAILABLE_GPU) -> torch.device:
    """Gets torch available device.

    Args:
        available_gpu: optional flag to use CUDA or CPU.

    Returns:
        CUDA device if available, else CPU device."""
    if available_gpu:
        return torch.device('cuda')
    return torch.device('cpu')


def to_gpu(
    tensor: Union[torch.Tensor, torch.nn.Module], available_gpu: bool = AVAILABLE_GPU
) -> Union[torch.Tensor, torch.nn.Module]:
    """Transfers tensor to GPU if available.

    Args:
        tensor: Tensor to transfer
        available_gpu: optional flag to use CUDA or CPU.

    Returns:
        Tensor transferred to CUDA if available, else CPU."""
    return tensor.cuda() if available_gpu else tensor.cpu()

def list_to_gpu(values: List, available_gpu: bool = AVAILABLE_GPU) -> List:
    """Transfers elements of a list to GPU if available. Needed for DINO
    
    Args:
        values: list to transfer
        available_gpu: optional flag to use CUDA or CPU.

    Returns:
        list transferred to CUDA if available, else CPU."""
    return [t.cuda() for t in values] if available_gpu else [t.cpu() for t in values]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
