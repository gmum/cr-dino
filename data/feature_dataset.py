from typing import List, Tuple
import gin
import pandas as pd
import torch

from torch.utils.data import Dataset

@gin.configurable
class FeatureDataset(Dataset):
    def __init__(self,
                 feature_file: str,
                 label_file: str,
                 feature_columns: List[str],
                 label_columns: List[str],
                 split: str,
                 mode: str,
                 info: bool = False,
                 image_id: str = 'image_id') -> None:
        """
        Dataset for handling pre-generated features.
        Args:
            feature_file: csv file with pre-generated features
            label_file: csv file with labels
            feature_columns: list of columns used in the feature_file corresponding to features
            label_columns: current labels, corresponds to the columns in the label_file
            split: current split in the train/test split
            mode: train/test
            info: if True, add additional information from rows in feature file
            image_id: name of a column containing id of image
        """
        super(FeatureDataset).__init__()
        self.features = pd.read_csv(feature_file)
        self.labels = pd.read_csv(label_file)
        image_ids = self.labels[self.labels[split] == mode].image_id.unique()
        self.features = self.features[self.features[image_id].isin(image_ids)]
        self.features.reset_index(inplace=True, drop=True)
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.image_id = image_id
        self.info = info

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        row = self.features.iloc[index]
        features = row[self.feature_columns].values
        row_labels = self.labels[self.labels[self.image_id] == row[self.image_id]]
        labels = row_labels[self.label_columns].values
        if self.info:
            infos = row[self.image_id]
            return torch.tensor(features.astype(float)), torch.tensor(labels.astype(int)), infos
        else:
            return torch.tensor(features.astype(float)), torch.tensor(labels.astype(int))
    
    @staticmethod
    def collate_fn(batch):
        """
        Collates data from dataset.
        Args: 
            batch: current batch
        Returns:
            tuple of tensor and tensor"""
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])
    

    @staticmethod
    def info_collate_fn(batch):
        """
        Collates data from dataset.
        Args: 
            batch: current batch
        Returns:
            tuple of tensor and tensor"""
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]