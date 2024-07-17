import gin
from typing import List
from torch import nn
import torch

@gin.configurable
class MLP(nn.Module):
    def __init__(self, num_features: int = 512, num_out: int = 30):
        super(MLP, self).__init__()
        self.num_features = num_features
        self.num_out = num_out
        self.layers = nn.Sequential(
                nn.Linear(self.num_features, 3 * 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(3 * 512, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return self.layers(x)
    

@gin.configurable
class MolMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               nn.ReLU())
                               )

        self.layers.append(nn.Sequential(
                           nn.Linear(hidden_dim, output_dim))
                           )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x