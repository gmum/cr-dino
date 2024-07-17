import sys

import gin
from torch.utils.data import DataLoader

from data.feature_dataset import FeatureDataset
from procedures.mlp_training import MLPTraining

"""
Runs cross-valdiation on 5 splits. 
Args:
    config_file: gin config
    batch_size: batch size
    model_name: base of model name, split number is added
"""


if __name__ == '__main__':
    config_file = sys.argv[1]
    batch_size = int(sys.argv[2])
    model_name = sys.argv[3]
    gin.parse_config_file(config_file)

    for i in range(5):
        train_dataset = FeatureDataset(mode='train', split='', feature_columns=[], label_columns=[])
        valid_dataset = FeatureDataset(mode='test', split='', feature_columns=[], label_columns=[])

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=FeatureDataset.collate_fn,
                                      num_workers=0)

        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=FeatureDataset.collate_fn,
                                      num_workers=0)

        tp = MLPTraining(model_name=f'{model_name}_{i}', label_columns=[])
        tp.run(train_data_loader=train_dataloader,
               test_data_loader=valid_dataloader)