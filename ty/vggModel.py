import torch.nn as nn
import torch.nn.functional as F
import math

class VGG(nn.Module):
    def __init__(self, features, num_classes=131, init_weights=True):
        super(VGG, self).__init__()
        self.feature = features
        self.classification = nn.Sequential(
            nn.Linear(512 * 12 * 12, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
             self._initialize_weights()

    def forward(self, x):
        y = self.feature(x)
        y = y.view(y.size(0), -1)
        y = self.classification(y)
        return y.float()


    def _initialize_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2./n))
             elif isinstance(m, nn.BatchNorm2d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()
             elif isinstance(m, nn.Linear):
                 m.bias.data.zero_()

    def make_layers(cfg, batch_norm=False):
        layer = []
        input_channel = 3
        # cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']
        for v in cfg:
            if v == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batch_norm:
                    layer += [nn.Conv2d(input_channel, v, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(v),
                        nn.ReLU()]
                else:
                    layer += [nn.Conv2d(input_channel, v, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()]
                input_channel = v

        return nn.Sequential(*layer)
