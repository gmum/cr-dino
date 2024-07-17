import pandas as pd
import numpy as np
import torch

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from torch.utils.data import Dataset

def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nbits, useChirality=chiral)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

class MolDataset(Dataset):
    def __init__(self, 
                 path: str,
                 image_id: str,
                 smiles: str) -> None:
        """
        Dataset for handling molecules. It read file with SMILES and returns Morgan fingerprints.
        Args:
            path: path to the file with compound SMILES
            image_id: identifiers' column in path file
            smiles: column containing SMILES
        """
        self.df = pd.read_csv(path)
        self.image_id = image_id
        self.smiles = smiles

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        morgan = morgan_from_smiles(row[self.smiles])
        return row[self.image_id], torch.tensor(morgan)
    