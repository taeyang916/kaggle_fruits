#%%
from __future__ import print_function, division
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image

class FruitDataLoader(DataLoader):
    def __init__(self, dataset_type='Training', transform=None):
        self.dataset = pd.read_csv(f'./{dataset_type}.csv')
        self.image_path = self.dataset['image_path']
        self.image_idx = self.dataset['image_idx']
        self.transform = transform


    def __getitem__(self, idx):
        x = Image.open(self.image_path[idx])
        y = self.image_idx[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)
