from typing import List
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset


class ImageMolDataset(Dataset):
    def __init__(self, 
                 img_path: str,
                 mol_path: str,
                 label_file: str,
                 split: str,
                 mode: str,
                 image_id: str,
                 image_feat_columns: List[str],
                 mol_feat_columns: List[str]) -> None:
        """
        Dataset for handling image and molecule features. To be used in the retrieval task.
        Args:
            img_path: csv file with pre-generated image features
            mol_path: csv file with pre-generated molecule features
            label_file: csv file with labels
            split: current split
            mode: train/test
            iamge_id: column name containing id of image
            image_feat_columns: list of columns containing features in img_path file
            mol_feat_columns: list of columns containing features in mol_path file
        """
        self.image_id = image_id
        self.img_features = image_feat_columns
        self.mol_features = mol_feat_columns
        self.labels = pd.read_csv(label_file)
        image_ids = self.labels[self.labels[split] == mode].image_id.unique()
        
        self.img_df = pd.read_csv(img_path)
        self.mol_df = pd.read_csv(mol_path)
        self.img_df = self.img_df[self.img_df[image_id].isin(image_ids)]
        self.mol_df = self.mol_df[self.mol_df[image_id].isin(self.img_df[image_id].unique())]
        self.img_df.reset_index(inplace=True, drop=True)
        self.mol_df.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.img_df)
    
    def __getitem__(self, index):
        row = self.img_df.iloc[index]
        curr_label = row[self.image_id]
        row2 = self.mol_df[self.mol_df[self.image_id] == curr_label].iloc[0]
        img_feat = np.array(row[self.img_features].values).astype(float)
        mol_feat = np.array(row2[self.mol_features].values).astype(float)
        return curr_label, torch.from_numpy(img_feat), torch.from_numpy(mol_feat)
    