import os
from typing import Callable

import gin
import pandas as pd
import torch
from utils.torch_utils import to_gpu
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn.vit_dino import vit_base
from utils.dino_utils import load_pretrained_weights


@gin.configurable
class DINOFeatureExtraction:
    def __init__(
        self,
        saving_path: str,
        model_path: str,
        features_file_name: str,
        dataset: Dataset,
        collate_fn: Callable,
        batch_size: int,
        num_workers: int,
        start_point: int = 0,
        mask_generator: Callable = None,
    ):
        """Procedure for DINO model feature extraction.

        Args:
            saving_path: Path where to save the extracted features
            model_path: Location of saved model
            features_file_name: Name of file with features
            dataset: dataset for extraction
            collate_fn: collate function
            batch_size: batch size
            num_workers: number of workers
            start_point: number of starting element from dataset (not iteration, batch size is accounted for)
            mask_generator: applies masking
        """

        print(f'Dataset length: {len(dataset)}')
        self.model = vit_base(apply_masking=True, mask_generator=mask_generator)
        load_pretrained_weights(self.model, model_path, 'teacher', "vit_base", 16)
        self.model = self.model.cuda()
        print(self.model)
        self.saving_path = saving_path
        self.features_file_name = features_file_name
        self.start_point = start_point // batch_size
        self.data_loader = DataLoader(
                dataset=dataset, 
                collate_fn=collate_fn, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
                )
        

        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def run(self):
        features = []
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(self.data_loader)

            for i in tqdm(range(self.start_point, len(self.data_loader)), desc='Batches'):
                image1, metadata = next(data_iter)
                latent_set = []
                for img_set in image1:
                    latent_set.append(torch.mean(self.model(to_gpu(img_set)), dim=0))
                for j, l in enumerate(latent_set):
                    features.append([*metadata[j], *[l1.detach().cpu().numpy() for l1 in l]])

                # print(features)

                if i == 0:
                    features = pd.DataFrame(data=features, columns=[*metadata[-1].index, *[f'tensor_{j} RAW_VALUE' for j in range(768)]])
                    features.to_csv(f'{self.saving_path}/{self.features_file_name}', index=False)
                    features = []

                elif i % 1000 == 0:
                    features = pd.DataFrame(data=features, columns=[*metadata[-1].index, *[f'tensor_{j} RAW_VALUE' for j in range(768)]])
                    features.to_csv(f'{self.saving_path}/{self.features_file_name}', mode='a', index=False, header=False)
                    features = []
            
            # Last save
            features = pd.DataFrame(data=features, columns=[*metadata[-1].index, *[f'tensor_{j} RAW_VALUE' for j in range(768)]])
            features.to_csv(f'{self.saving_path}/{self.features_file_name}', mode='a', index=False, header=False)