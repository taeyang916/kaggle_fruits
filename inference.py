import torch.nn as nn
import torch
import torchvision
import albumentations
import albumentations.pytorch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import MyModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = MyModel()
model.load_state_dict(torch.load('./checkpoint.pth'))
model.eval()
model.to(device)


img_list = ['./cat.jpg', './dog.jpg']

test_transfrom = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2()
])

for img in img_list:
    img = Image.open(img)
    img = test_transfrom(image=np.array(img))['image']
    with torch.no_grad():
        img = img.reshape(1, 3, 224, 224).float().to(device)
        output = model(img)
        prediction = torch.argmax(output, -1)
        if prediction.item() == 0:
            print('cat')
        else:
            print('dog')