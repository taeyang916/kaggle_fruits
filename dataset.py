import os
import numpy as np
from glob import glob

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms, utils

class CatDogDataset(Dataset):
    def __init__(self, root_dir='./data', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = glob(os.path.join(root_dir, '*.jpg'))


    def set_transform(self, transform):
        if self.transform:
            self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        label = 0 if 'cat' in self.img_list[idx] else 1

        if self.transform:
            img = self.transform(image=np.array(img))['image']
        
        return img, label