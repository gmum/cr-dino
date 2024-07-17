from typing import OrderedDict

from tqdm import tqdm
from data.mol_dataset import MolDataset
from nn.mlp import MolMLP

import pandas as pd
import torch

def key_transformation(x):
    return x.replace('module.transformer.', '')

def get_model(model, path):
    old_State_dict = torch.load(path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()

    for key, value in old_State_dict['state_dict'].items():
        if key.startswith('module.transformer'):
            new_key = key_transformation(key)
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

if __name__ == '__main__':
    output_file = ''
    mol = MolMLP(input_dim =  1024,
                 hidden_dim = 1024,
                 output_dim = 512,
                 n_layers = 4)
    
    model = get_model(mol, '').cuda()
    model.eval()
    dataset = MolDataset()
    data_iter = iter(dataset)
    features = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc='Molecules'):
            label, morgan = next(data_iter)
            print(morgan.dtype)
            enc = model(morgan.float().cuda().unsqueeze(0))
            features.append([label, *[e for e in enc.squeeze().detach().cpu().numpy()]])

            if i == 0:
                features = pd.DataFrame(data=features, columns=['image_id', *[f'tensor_{j}' for j in range(512)]])
                features.to_csv(output_file, index=False)
                features = []

            elif i % 1000 == 0:
                features = pd.DataFrame(data=features, columns=['image_id', *[f'tensor_{j}' for j in range(512)]])
                features.to_csv(output_file, mode='a', index=False, header=False)
                features = []
        
        # Last save
        features = pd.DataFrame(data=features, columns=['image_id', *[f'tensor_{j}' for j in range(512)]])
        features.to_csv(output_file, mode='a', index=False, header=False)