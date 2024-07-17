import torch
import torchvision

from gin import config

# torch optim
TORCH_OPTIM = 'torch.optim'
torch.optim.AdamW = config.external_configurable(torch.optim.AdamW, module=TORCH_OPTIM)
torch.optim.Adam = config.external_configurable(torch.optim.Adam, module=TORCH_OPTIM)

TORCHVISION_TRANSFORMS = 'torchvision.transforms'
config.external_configurable(torchvision.transforms.ColorJitter, module=TORCHVISION_TRANSFORMS)
config.external_configurable(torchvision.transforms.Resize, module=TORCHVISION_TRANSFORMS)
config.external_configurable(torchvision.transforms.RandomCrop, module=TORCHVISION_TRANSFORMS)