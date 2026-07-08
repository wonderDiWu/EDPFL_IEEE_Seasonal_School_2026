'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg8', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.pps = features
        self.pps.append(nn.Sequential(
            Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, num_classes, bias=False),
        ))
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()


    def forward(self, x):
        for pp in self.pps:
            x = pp(x)
        return x


def make_layers(cfg, batch_norm=False):
    pps = nn.ModuleList()
    in_channels = 3
    for pp in cfg:
        layers = []
        for v in pp:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pps.append(nn.Sequential(*layers))
    return pps


cfg = {
    'A': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']]
}

def vgg11(num_classes):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes)


def vgg11_bn(num_classes):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)