import torch
import torch.nn as nn
import sys
import copy
sys.path.append("../")
import utils

class Partitioner(torch.nn.Module):
    def __init__(self, model, pp, side):
        super(Partitioner, self).__init__()
        # Validation check
        if not (0 < pp < len(model.pps)):
            if pp != -1:
                raise ValueError('Invalid pp!')
        if pp == -1: # We use -1 to denote device native training
            pp = len(model.pps)

        self.pp = pp
        self.side = side
        self.pps = [nn.Sequential()] * len(model.pps)
        if side == 'Device':
            self.pps[:pp] = list(copy.deepcopy(model.pps[:pp]))
        if side == 'Server':
            self.pps[pp:] = list(copy.deepcopy(model.pps[pp:]))
        self.pps = nn.ModuleList(self.pps)

    def forward(self, x):
        for l in self.pps:
            x = l(x)
        return x

def unit_test():
    lenet = utils.get_model('LeNet', 10, 'cpu')
    print(lenet)
    input_data = torch.randn((10,3,32,32))
    print(lenet(input_data).size())
    
    pps = range(1,5)
    for pp in pps:
        print(pp)
        device_model = Partitioner(lenet, pp, 'Device')
        print(device_model)
        device_output = device_model(input_data)
        print(device_output.size())

        server_model = Partitioner(lenet, pp, 'Server')
        print(server_model)
        print(server_model(device_output).size())
    
    print('-'*100)
    vgg11 = utils.get_model('VGG11', 10, 'cpu')
    print(vgg11)
    input_data = torch.randn((10,3,32,32))
    print(vgg11(input_data).size())
    
    pps = range(1,6)
    for pp in pps:
        print(pp)
        device_model = Partitioner(vgg11, pp, 'Device')
        print(device_model)
        device_output = device_model(input_data)
        print(device_output.size())

        server_model = Partitioner(vgg11, pp, 'Server')
        print(server_model)
        print(server_model(device_output).size())


    print('-'*100)
    resnet12 = utils.get_model('ResNet12', 10, 'cpu')
    print(resnet12)
    input_data = torch.randn((10,3,32,32))
    print(resnet12(input_data).size())
    
    pps = range(1,6)
    for pp in pps:
        print(pp)
        device_model = Partitioner(resnet12, pp, 'Device')
        print(device_model)
        device_output = device_model(input_data)
        print(device_output.size())

        server_model = Partitioner(resnet12, pp, 'Server')
        print(server_model)
        print(server_model(device_output).size())
    
    return True

if __name__ == '__main__':
    if unit_test():
        print('Unit test pass!')

