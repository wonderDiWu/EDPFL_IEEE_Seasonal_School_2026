import torch.nn as nn
import collections

class Global_Freezer:
    def __init__(self, fedfreeze_config):
        self.fedfreeze_config = fedfreeze_config
        self.thres1 = fedfreeze_config['Global_Freezer']['thres1']
        self.thres2 = fedfreeze_config['Global_Freezer']['thres2']

    def build(self, uninet):
        self.layers_dict_with_types = self.get_layers_with_types(uninet)
        self.frozen_dict = self.get_frozen_dict(uninet.state_dict().keys())

    def get_layers_with_types(self, model):
        layers_dict_with_types = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'Conv2d' not in layers_dict_with_types:
                    layers_dict_with_types['Conv2d'] = []
                layers_dict_with_types['Conv2d'].append(name)
            elif isinstance(module, nn.BatchNorm2d):
                if 'BatchNorm2d' not in layers_dict_with_types:
                    layers_dict_with_types['BatchNorm2d'] = []
                layers_dict_with_types['BatchNorm2d'].append(name)
            elif isinstance(module, nn.Linear):
                if 'Linear' not in layers_dict_with_types:
                    layers_dict_with_types['Linear'] = []
                layers_dict_with_types['Linear'].append(name)
            else:
                pass
        return layers_dict_with_types
    
    def get_frozen_dict(self, layer_dict):
        ordered_layer_dict = collections.OrderedDict()
        for l in layer_dict:
            ordered_layer_dict[l] = 0
        return ordered_layer_dict
    
    def global_freezing(self, norm_eps, rate_eps):
        last_layer = list(self.frozen_dict.keys())[-1]
        for k in self.frozen_dict:
            if k.replace('.weight','').replace('.bias','') in self.layers_dict_with_types['Conv2d'] or k.replace('.weight','').replace('.bias','') in self.layers_dict_with_types['Linear']:
                # Frozen rules
                if k != last_layer: 
                    if self.frozen_dict[k] == 1:
                        continue
                    else:
                        if norm_eps[k] <= self.thres1 and abs(rate_eps[k]) <= self.thres2:
                            self.frozen_dict[k] = 1
                        if k == 'pps.0.0.weight':
                            self.frozen_dict[k] = 1
                        break
        
        # Freezing corresponding BatchNorm2d layers
        for k in self.frozen_dict:
            layer_prefix = k.replace('.weight','').replace('.bias','')
            if layer_prefix in self.layers_dict_with_types['Conv2d'] and self.frozen_dict[k] == 1:
                if 'BatchNorm2d' in self.layers_dict_with_types.keys():
                    batch_norm_prefix = self.layers_dict_with_types['BatchNorm2d'][self.layers_dict_with_types['Conv2d'].index(layer_prefix)]
                    batch_norm_w = batch_norm_prefix + '.weight'
                    batch_norm_b = batch_norm_prefix + '.bias'
                    if batch_norm_w in self.frozen_dict.keys() and batch_norm_b in self.frozen_dict.keys():
                        self.frozen_dict[batch_norm_w] = 1
                        self.frozen_dict[batch_norm_b] = 1 

    def freeze_layers_by_names(self, net):
        for name, param in net.named_parameters():
            if self.frozen_dict[name] == 1:
                param.requires_grad = False