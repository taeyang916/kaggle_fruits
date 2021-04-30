import torch
import torchvision
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


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)

    classes = Labelling.labelling_()

    # cfg_type : 'personal_net', 'vgg16'
    cfg = Configure.make_configure(cfg_type='personal_net')

    transform = tfm.create_transformer()
    batchsize = 10

    test_dataset = FruitDataLoader(dataset_type='Test', transform=transform)
    testloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    model_path = os.getcwd() + '/ty/model.pth'
    vgg16 = torch.load(model_path)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = vgg16(images.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = vgg16(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Acc : %d %%' % (100*correct / total))

test()