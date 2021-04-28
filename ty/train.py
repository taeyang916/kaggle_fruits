#%%#
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from vggModel import VGG
from dataset import FruitDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    torch.manual_seed(123)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']
    epochs = 30


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    ])

    # dataset_type : 'Training' or 'Test'
    train_dataset = FruitDataLoader(dataset_type='Training', transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

    test_dataset = FruitDataLoader(dataset_type='Test', transform=transform)
    testloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    vgg16 = VGG(VGG.make_layers(cfg), 131, True).to(device)

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

if __name__ == '__main__':
    main()


"""
device cuda or cpu
vgg net
"""
# %%
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




for train_batch in trainloader:
    image, label = train_batch
    for i in range(3):
        plt.imshow(image[i].permute(1, 2, 0))
        plt.show()
        print(label[i])
    break

"""