from typing import Any

import numpy as np
import gin
import torch


@gin.configurable
class NoMask:
    def __init__(self) -> None:
        pass

    def __call__(self) -> Any:
        return torch.tensor([0,0, 0, 0 ,0], dtype=bool)
    

@gin.configurable
class RandomChannelMask:
    def __init__(self, max_ch, min_ch) -> None:
        self.max_ch = max_ch
        self.min_ch = min_ch
        self.all_channels = np.array([x for x in range(max_ch)])

    def __call__(self) -> Any:
        no_channels = np.random.randint(self.min_ch, self.max_ch, 1)
        channels = np.random.choice(self.all_channels, size=no_channels, replace=False)
        x = torch.zeros(self.max_ch, dtype=bool)
        x[channels] = 1
        return x
    
@gin.configurable
class ScheduleChannelMask:
    def __init__(self, max_ch, min_ch, total_it) -> None:
        self.max_ch = max_ch
        self.min_ch = min_ch
        self.all_channels = np.array([x for x in range(max_ch)])
        self.interval = total_it//(self.max_ch - self.min_ch + 1) if total_it > 0 else 0

    def __call__(self, x) -> Any:
        over_interval = x % self.interval
        no_channels = self.max_ch - (x - over_interval) // self.interval
        channels = np.random.choice(self.all_channels, size=no_channels, replace=False)
        x = torch.zeros(self.max_ch, dtype=bool)
        x[channels] = 1
        return x

@gin.configurable
class SetMask:
    def __init__(self, max_ch, channels) -> None:
        self.channels = np.array(channels)
        self.max_ch = max_ch

    def __call__(self) -> Any:
        x = torch.zeros(self.max_ch, dtype=bool)
        x[self.channels] = 1
        return x