import torch
import torch.nn as nn
import torch.nn.functional as func

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)
    
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.pps = nn.ModuleList()
        self.pps.append(nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)))
        self.pps.append(nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)))
        self.pps.append(nn.Sequential(Flatten(), nn.Linear(16*5*5, 120, bias=False), nn.ReLU(inplace=True)))
        self.pps.append(nn.Sequential(nn.Linear(120, 84, bias=False), nn.ReLU(inplace=True)))
        self.pps.append(nn.Linear(84, num_classes, bias=False))

    def forward(self, x):
        for pp in self.pps:
            x = pp(x)
        return x