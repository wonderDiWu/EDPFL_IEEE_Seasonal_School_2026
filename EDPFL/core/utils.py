# Some helper functions and classes for EDPFL

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import pickle, struct, socket
from model import lenet, vgg, resnet, partitioner
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model(model_name, num_classes, device):
    if model_name == 'LeNet':
        net = lenet.LeNet(num_classes)
    if model_name == 'VGG11':
        net = vgg.vgg11_bn(num_classes)
    if model_name == 'ResNet12':
        net = resnet.resnet12(num_classes)
    net = net.to(device)
    return net

def get_partition_model(model_name, num_classes, device, pp, side):
    if model_name == 'LeNet':
        net = lenet.LeNet(num_classes)
    if model_name == 'VGG11':
        net = vgg.vgg11_bn(num_classes)
    if model_name == 'ResNet12':
        net = resnet.resnet12(num_classes)
    partition_net = partitioner.Partitioner(net, pp, side)
    partition_net = partition_net.to(device)
    return partition_net

def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        if isinstance(m, nn.ConvTranspose2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            #init.zeros_(m.running_mean)
            #init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net

def fed_avg(aggregrated_model, w_local_list):
    keys = w_local_list[0][0].keys()
    for k in keys:
        for w in w_local_list:
            beta = w[1]
            if 'num_batches_tracked' in k:
                aggregrated_model[k] = w[0][k]
            else:	
                aggregrated_model[k] += (w[0][k] * beta)
    return aggregrated_model

def transfer_weights_holistic(weights_file,weights):
    pretrained_weights = torch.load(weights_file, map_location=torch.device('cpu'))
    for key in weights:
        if len(weights[key].size()) != 2: # Exclude the fully connected layers
            assert weights[key].size() == pretrained_weights[key].size()
            weights[key] = pretrained_weights[key]
    return weights

def adjust_lr(r, lr, decay, lr_schedule):
    lr_schedule = lr_schedule.split('-')
    for i in range(len(lr_schedule)):
        if r > int(lr_schedule[i]):
            pass
        else:
            return lr * ( decay ** i)
    return lr * ( decay ** len(lr_schedule))

def move_state_dict_to_device(state_dict, device):
    # Create an empty state dictionary on the GPU
    state_dict_on_device = {}
    
    # Move each weight to the GPU and add it to the new state dictionary
    for key in state_dict:
        state_dict_on_device[key] = state_dict[key].to(device)
    
    return state_dict_on_device

def split_weights(weights, pweights):
    for key in pweights:
        assert pweights[key].size() == weights[key].size()
        pweights[key] = weights[key]
    return pweights

def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()

    for key in cweights:
        assert cweights[key].size() == weights[key].size()
        concat_dict[key] = cweights[key]

    for key in sweights:
        assert sweights[key].size() == weights[key].size()
        concat_dict[key] = sweights[key]

    return concat_dict

def check_frozen_dict_device(device_weights_dict, frozen_dict, current_frozen_params):
    last_frozen_layer = None
    if len(current_frozen_params) > 0:
        current_last_frozen_layer = list(current_frozen_params.keys())[0]
    else:
        current_last_frozen_layer = None
        return False, current_last_frozen_layer

    for k in device_weights_dict.keys():
        # We set the last frozen layer as the key index
            if frozen_dict[k] == 1:
                last_frozen_layer = k
            else:
                break
    if last_frozen_layer == current_last_frozen_layer:
        return True, current_last_frozen_layer
    else:
        return False, current_last_frozen_layer

def concat_frozen_weights(weights, local_frozen_weights, truncated_init_cweights):
    concat_dict = collections.OrderedDict()

    for key in local_frozen_weights:
        assert local_frozen_weights[key].size() == weights[key].size()
        concat_dict[key] = local_frozen_weights[key]

    for key in truncated_init_cweights:
        assert truncated_init_cweights[key].size() == weights[key].size()
        concat_dict[key] = truncated_init_cweights[key]

    assert len(concat_dict) == len(weights)
    return concat_dict

def save_current_frozen_params(weights, frozen_dict):
    last_frozen_layer = None
    local_frozen_weights = collections.OrderedDict()
    for k in weights:
        if frozen_dict[k] == 1:
            local_frozen_weights[k] = weights[k]
            last_frozen_layer = k
        else:
            break
    return last_frozen_layer, local_frozen_weights

def concat_frozen_updated_weights(weights, init_weights, truncated_updated_weights):
    concat_dict = collections.OrderedDict()
    # This will only be used when pp == -1
    assert len(weights) == len(init_weights)
    for key in init_weights:
        if key in truncated_updated_weights.keys():
            concat_dict[key] = truncated_updated_weights[key]
        else:
            concat_dict[key] = init_weights[key]
    return concat_dict

def unit_test():
    vgg11 = get_model('VGG11', 10, 'cpu')
    pp = 1
    device_model = partitioner.Partitioner(vgg11, pp, 'Device')
    server_model = partitioner.Partitioner(vgg11, pp, 'Server')
    full_keys = list(vgg11.state_dict().keys())
    d_keys = list(device_model.state_dict().keys())
    s_keys = list(server_model.state_dict().keys())
    print(full_keys)
    print(d_keys)
    print(s_keys)
    for param in vgg11.parameters():
        param.data.fill_(1)
    for param in device_model.parameters():
        param.data.fill_(2)
    for param in server_model.parameters():
        param.data.fill_(3)
    
    #print(vgg11.state_dict()[full_keys[0]], device_model.state_dict()[d_keys[0]], server_model.state_dict()[s_keys[0]])
    cweights = split_weights(vgg11.state_dict(), device_model.state_dict())
    sweights = split_weights(vgg11.state_dict(), server_model.state_dict())
    server_model.load_state_dict(sweights)
    #print(vgg11.state_dict()[full_keys[0]], device_model.state_dict()[d_keys[0]], server_model.state_dict()[s_keys[0]])
    print(sweights[s_keys[0]])
    print(concat_weights(vgg11.state_dict(), device_model.state_dict(), sweights)[full_keys[0]])
    
    

    return True

if __name__ == '__main__':
     if unit_test():
          print('Unit test pass!')
