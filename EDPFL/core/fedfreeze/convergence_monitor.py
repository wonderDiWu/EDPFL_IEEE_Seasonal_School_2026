# FedFreeze Convergence_Monitor
import sys
import torch
import copy
import numpy as np

from collections import OrderedDict

class Convergence_Monitor():
    def __init__(self, fedfreeze_config):
        self.alpha = fedfreeze_config['Convergence_Monitor']['alpha']
        self.delta, self.pre_delta = OrderedDict(), OrderedDict()
        self.eps = OrderedDict()
        self.norm_eps = OrderedDict()
        self.rate_eps = OrderedDict()
        self.pre_eps = OrderedDict()
        self.init_eps = OrderedDict()

    def build(self, init_weight, layers_dict_with_types):
         self.layer_keys = init_weight.keys()
         self.pre_weight = copy.deepcopy(init_weight)
         self.layers_dict_with_types = layers_dict_with_types

    def check_layer_convergence(self, new_weight):
        assert len(self.pre_weight) == len(new_weight.keys())
        for k in self.pre_weight:
            # We only monitor Conv2d and Linear Layer
            if k.replace('.weight','').replace('.bias','') in self.layers_dict_with_types['Conv2d'] or k.replace('.weight','').replace('.bias','') in self.layers_dict_with_types['Linear']:
                layer_ep = torch.abs(new_weight[k] - self.pre_weight[k])
                if k in self.eps.keys():
                    self.pre_eps[k] = copy.deepcopy(self.eps[k])
                    self.eps[k] = self.eps[k] * self.alpha + ( 1 - self.alpha) * torch.mean(layer_ep.view(-1).float()).item()
                else:
                    self.eps[k] = torch.mean(layer_ep.view(-1).float()).item()
                    self.pre_eps[k] = copy.deepcopy(self.eps[k])
                    self.init_eps[k] = copy.deepcopy(self.eps[k])
                self.norm_eps[k] = self.eps[k] / self.init_eps[k]

                if k in self.rate_eps.keys():
                    rate = (self.pre_eps[k] - self.eps[k]) / self.pre_eps[k]
                    self.rate_eps[k] = self.rate_eps[k] * self.alpha + ( 1 - self.alpha) * rate
                else:
                    rate = (self.pre_eps[k] - self.eps[k]) / self.pre_eps[k]
                    self.rate_eps[k] = rate           
        self.pre_weight = copy.deepcopy(new_weight)
        return self.norm_eps, self.rate_eps
            
