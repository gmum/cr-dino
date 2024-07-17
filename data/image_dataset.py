from typing import List, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import default_collate
from torchvision.transforms import Compose, Normalize


@gin.configurable
class CellPaintDataset(Dataset):
    def __init__(self, 
                 metadata_file: str,
                 norm_file_mean: Optional[str],
                 norm_file_std: Optional[str],
                 dataset_base: str, 
                 compound_column: str = '',
                 plate_column: str = '',
                 well_column: str = '',
                 site_column: str = '',
                 transforms: List = [],
                 control: bool = False,
                 row_info: bool = False,
                 ):
        """
        Dataset for handling images in Bray et al. dataset. Images are saved in the .pt format as tensors.
        
        Args:
            metadata_file: file containg dataset metadata
            norm_file_mean, norm_file_std: files containing values for normalization - mean and std for each channel
            dataset_base: path for the dataset up to well level
            compound_column: columns with compound identification
            plate_column: column with plate identification (5 digit number)
            well_column: column with well identification (a01-p24)
            site_column: column with site identification (s1-s6)
            transforms: list of transforms to perform on the image
            control: if True use DMSO to create dataset otherwise do not use DMSO
            row_info: if True, return row information
        """
        self.dataset_base = dataset_base
        self.metadata = pd.read_csv(metadata_file)
        if control:
            self.metadata = self.metadata[self.metadata[compound_column] == 'DMSO']
        else:
            self.metadata = self.metadata[self.metadata[compound_column] != 'DMSO']
        self.row_info = row_info
        self.metadata[plate_column] = self.metadata[plate_column].astype(int)
        self.norm_file = norm_file_mean
        transforms_base = []
        if self.norm_file is not None:
            norm_values_mean = np.load(norm_file_mean, allow_pickle=True).item()
            norm_values_std = np.load(norm_file_std, allow_pickle=True).item()
            self.norm_file = norm_values_mean, norm_values_std
        else:
            print('Using static normalization')
            # Input stats for the dataset. For HCS, calculate using DMSO wells.
            norm = Normalize(mean=[], 
                             std= [])
            transforms_base = [norm]
        if len(transforms) > 0:
            transforms_base.extend(transforms)
        self.transforms = Compose(transforms_base)
        self.compound_column = compound_column
        self.compounds = self.metadata[compound_column].unique()
        self.plate_column = plate_column
        self.well_column = well_column
        self.site_column = site_column


    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[str]]:
        row = self.metadata.iloc[idx]
        img = torch.load(f'{self.dataset_base}/{row[self.plate_column]}/{row[self.well_column]}_{row[self.site_column]}.pt')
        img = img.float()
        if self.norm_file:
            norm = Normalize(self.norm_file[0][row[self.plate_column]], self.norm_file[1][row[self.plate_column]])
            img = norm(img)
        img = self.transforms(img)
        if self.row_info:
            return img, row
        else:
            return img

@gin.configurable
def info_collate_fn(batch):
    """
    Collates data from dataset.
    Args: 
        batch: current batch
    Returns:
        tuple of tensor and list"""
    imgs = default_collate([b[0] for b in batch])
    info = [b[1] for b in batch]
    return imgs, info

@gin.configurable
def basic_collate_fn(batch):
    """
    Collates data from dataset.
    Args: 
        batch: current batch
    Returns:
        tuple of tensor and list"""
    print('collate', batch[0].shape)
    return torch.stack([b[0] for b in batch])
