from albumentations.augmentations.transforms import Normalize

import torch
import torchvision
import albumentations
import albumentations.pytorch
from adamp import AdamP
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary as summary_

from dataset import CatDogDataset
from model import MyModel

NUM_EPOCHS = 3
BATCH_SIZE = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_transfrom = albumentations.Compose([    
    albumentations.OneOf([        
        albumentations.HorizontalFlip()
    ]),
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2()
])

dataset = CatDogDataset(transform=train_transfrom)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = MyModel()
model.to(device)
summary_(model, (3, 224, 224), batch_size=128)

optimizer = AdamP(model.parameters(), lr=1e-3, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)
criterion = nn.CrossEntropyLoss()


for epoch in range(NUM_EPOCHS):    
    model.train()
    loss_val = 0
    matches = 0
    for idx, train_batch in enumerate(train_loader):
        img, label = train_batch
        img = img.float().to(device)
        label = label.long().to(device)

        logit = model(img)
        loss = criterion(logit, label)
        loss_val += loss.item()

        pred = torch.argmax(logit, -1)
        matches += (pred == label).sum().item()

        loss_val /= BATCH_SIZE
        matches /= BATCH_SIZE
    
        optimizer.zero_grad()

        print(f'Epoch : {epoch + 1}/{NUM_EPOCHS} ({idx + 1}/{len(train_loader)})\n'
              f'Loss : {loss_val:.4f}\n'
              f'Accuracy : {matches:.4f}\n'
              f'cat Label : {label.tolist().count(0)}\n'
              f'cat Pred : {pred.tolist().count(0)}\n'
              f'dog Label : {label.tolist().count(1)}\n'
              f'dog Pred : {pred.tolist().count(1)}\n')
        loss_val = 0
        matches = 0        
        loss.backward()
        optimizer.step()
        scheduler.step()

torch.save(model.state_dict(), './checkpoint.pth')
        

        
