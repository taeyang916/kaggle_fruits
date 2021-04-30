import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from vggModel import VGG
from labelling import Labelling
from dataset import FruitDataLoader
from transformer import Transformer as tfm
from configure import Configure



def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)

    classes = Labelling.labelling_()

    # cfg_type : 'personal_net', 'vgg16'
    cfg = Configure.make_configure(cfg_type='personal_net')

    epochs = 30
    transform = tfm.create_transformer()
    batchsize = 16

    # dataset_type : 'Training' or 'Test'
    train_dataset = FruitDataLoader(dataset_type='Training', transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    vgg16 = VGG(VGG.make_layers(cfg), len(classes), True).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.005, momentum=0.9)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    print('Train Dataset : %d' % len(trainloader))
    print("====================================================================================")
    vgg16.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (input, target) in enumerate(trainloader, 0):
            inputs = input.to(device)
            labels = target.to(device)
            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/30))
                running_loss = 0.0
        lr_sche.step()
    print("====================================================================================")

    # training model 저장
    save_path = os.getcwd() + '/ty/model.pth' 
    print(save_path)
    torch.save(vgg16, save_path)

train()