#%%
from __future__ import print_function, division
import os
import torch
import csv
import pandas as pd
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, utils
import torchvision

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

"""
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
])

# dataset_type : 'Training' or 'Test'
train_dataset = FruitDataLoader(dataset_type='Training', transform=transform)
trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = FruitDataLoader(dataset_type='Test', transform=transform)
testloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
"""