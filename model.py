import torch.nn as nn
import timm

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model('resnet18d', pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.net(x)

        return x
