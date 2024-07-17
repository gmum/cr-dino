from typing import Any, List, Union, Iterable

import pandas as pd
from torch import Tensor

import gin
import torch
import numpy as np
from PIL import ImageFilter, ImageOps
import random
from torchvision import transforms
from PIL import Image


@gin.configurable
class SampleChannels:
    """
    Sample number of classes or schedule
    """
    def __init__(self, 
                 total_epochs: int, 
                 decrease: bool = True, 
                 max_ch: int = 5, 
                 min_ch: int = 2) -> None:
        """
        Args:
        total_it: number of total epochs. Ignored if decrease is False
        decrease: if True sets a decreasing schedulre, if False number of channels is random.
        max_ch: max number of channels
        min_ch: min number of channels (has to be bigger than 0)
        """
        assert min_ch > 0, "min_ch should be bigger than 0 - be realistic"
        self.total_epochs = total_epochs if decrease else 0
        self.max_ch = max_ch
        self.min_ch = min_ch
        self.interval = total_epochs//(self.max_ch - self.min_ch + 1) if decrease else 0
        self.all_channels = np.array([x for x in range(self.max_ch)])


    def __call__(self, x: int) -> List[int]:
        if self.total_epochs == 0:
            curr_step = np.random.randint(self.min_ch, self.max_ch, 1)
        else:
            over_interval = x % self.interval
            curr_step = (x - over_interval) // self.interval
        channels = np.random.choice(self.all_channels, size=curr_step, replace=False)
        return channels


class SelectChannels:
    def __init__(self) -> None:
        pass
    def __call__(self, imgs, channels) -> Any:
        return imgs[channels, :, :]
    
@gin.configurable
class ZeroChannels:
    def __init__(self, channels) -> None:
        self.channels = channels

    def __call__(self, imgs) -> Any:
        imgs[self.channels, :, :] = 0.
        return imgs
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

@gin.configurable
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flips = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            flips,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            flips,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            flips,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
@gin.configurable()
class RepeatChannels:
    def __call__(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__

@gin.configurable()
class FiveCropStack:
    def __init__(self, size: Union[int, Iterable[int]]):
        self.crop = transforms.FiveCrop(size)

    def __call__(self, x: Tensor) -> Tensor:
        patches = self.crop(x)

        # return torch.stack(patches)
        return patches


    
@gin.configurable
class ZeroRandomChannels:
    def __init__(self, min_ch=2, max_ch=5) -> None:
        self.max_ch = max_ch
        self.min_ch = min_ch
        self.all_channels = np.array([x for x in range(max_ch)])

    def __call__(self, imgs) -> Any:
        no_channels = np.random.randint(0, self.max_ch - self.min_ch, 1)
        channels = np.random.choice(self.all_channels, size=no_channels, replace=False)
        imgs[channels, :, :] = 0.
        return imgs
    
@gin.configurable
class ZeroScheduleChannels:
    def __init__(self, min_ch=2, max_ch=5, total_it=101) -> None:
        self.max_ch = max_ch
        self.min_ch = min_ch
        self.all_channels = np.array([x for x in range(max_ch)])
        self.interval = total_it//(self.max_ch - self.min_ch + 1) if total_it > 0 else 0
        print(self.interval)

    def __call__(self, imgs, x) -> Any:
        no_channels = x  // self.interval
        channels = np.random.choice(self.all_channels, size=no_channels, replace=False)
        for i in range(len(imgs)):
            imgs[i][channels, :, :] = 0.
        return imgs