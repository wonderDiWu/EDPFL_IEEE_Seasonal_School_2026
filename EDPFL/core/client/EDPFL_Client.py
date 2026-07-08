# FL Client class
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
import time
from tqdm import tqdm
import random
import copy
import collections

import utils
from communicator import *
from .FL_Client import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDPFL_Client(FL_Client):
    def __init__(self, server_addr, server_port, ip, index, device, model, pp, num_classes, FedAdapt_Modules, EcoFed_Modules, FedFreeze_Modules):
        self.pp = pp
        self.FedAdapt_Modules = FedAdapt_Modules
        self.EcoFed_Modules = EcoFed_Modules
        self.FedFreeze_Modules = FedFreeze_Modules
        super(EDPFL_Client, self).__init__(server_addr, server_port, ip, index, device, model)

        # FedAdapt pre_profiling
        model = utils.get_partition_model(self.model, num_classes, self.device, -1, 'Device')
        observation = self.FedAdapt_Modules.preprocessor.observe(model, self.device, num_classes)
        # Sending observation to server
        msg = ['MSG_OBSERVATION', observation]
        self.communicator.send_msg(self.communicator.sock, msg)

    def initialize(self, lr, num_classes):
        self.lr = lr
        # FedAdapt observation
        model = utils.get_partition_model(self.model, num_classes, self.device, -1, 'Device')
        observation = self.FedAdapt_Modules.preprocessor.observe(model, self.device, num_classes)
        # Sending observation to server
        msg = ['MSG_OBSERVATION', observation]
        self.communicator.send_msg(self.communicator.sock, msg)

        # Receiving partitioning point from server
        msg = self.communicator.recv_msg(self.communicator.sock)
        self.pp = msg[1]

        logger.debug('Building Model.')
        self.net = utils.get_partition_model(self.model, num_classes, self.device, self.pp, 'Device') # Get partition network
        logger.debug(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                      momentum=0.9, weight_decay=5e-4)

        logger.debug('Receiving Freezing Configs..')
        msg = self.communicator.recv_msg(self.communicator.sock)
        # Global freezing
        self.FedFreeze_Modules.global_freezer.frozen_dict = msg[1]
        # local freezing
        self.FedFreeze_Modules.layer_regularizer.adaptive_mu_factor = msg[2]

        # Sending back whether there is a saved frozen dict locally
        local_loading_flag, current_last_frozen_layer = utils.check_frozen_dict_device(self.net.state_dict(), self.FedFreeze_Modules.global_freezer.frozen_dict, self.FedFreeze_Modules.current_frozen_params)
        self.communicator.send_msg(self.communicator.sock, ['MSG_LOCAL_LOADING_FLAG', local_loading_flag])

        logger.debug('Receiving Global Weights..')
        msg = self.communicator.recv_msg(self.communicator.sock)
        self.communicator.send_msg(self.communicator.sock, ['MSG_TIME_RECORD'])
        if local_loading_flag:
            weights = utils.concat_frozen_weights(self.net.state_dict(), self.FedFreeze_Modules.current_frozen_params[current_last_frozen_layer], msg[1])
        else:
            weights = msg[1]
            last_frozen_layer, local_frozen_weights = utils.save_current_frozen_params(weights, self.FedFreeze_Modules.global_freezer.frozen_dict)
            self.FedFreeze_Modules.current_frozen_params = {}
            self.FedFreeze_Modules.current_frozen_params[last_frozen_layer] = local_frozen_weights
        self.net.load_state_dict(weights)

        # Saving init net for local freezing
        self.init_net = copy.deepcopy(self.net)
        logger.debug('Initialize Finished')

    def train(self, dataloader, E):
        # Training start
        time_acv_comm = 0
        training_loss = 0
        id, trainloader = dataloader[0], dataloader[1]
        self.net.to(self.device)

        if self.pp == -1:
            self.net.train()
            # Global freezer
            self.FedFreeze_Modules.global_freezer.freeze_layers_by_names(self.net)
            # Local freezer
            self.FedFreeze_Modules.local_freezer.build(T = E * len(trainloader))
        else:
            # Receiving activation switch flag
            msg = self.communicator.recv_msg(self.communicator.sock)
            update_flag = msg[1]

            self.net.eval()

        if self.pp == -1: # No offloading training
            for e in range(E):
                logger.debug('Epoch: {:}'.format(E))
                for batch_id, (inputs, targets) in enumerate(tqdm(trainloader)):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    #training_loss += loss.item()
                    loss.backward()
                    clip_grad_norm_(self.net.parameters(), max_norm=10)
                    self.optimizer.step()

                    # Local freezing
                    cur_T = e * len(trainloader) + batch_id
                    if self.FedFreeze_Modules.local_freezer.ask(cur_T):
                        if len(self.FedFreeze_Modules.local_freezer.T_list) == 0:
                            self.FedFreeze_Modules.local_freezer.T_list = self.FedFreeze_Modules.layer_regularizer.generate_T_list(self.init_net, self.net, self.FedFreeze_Modules.local_freezer.T)
                            logger.info(self.FedFreeze_Modules.local_freezer.T_list)
                        self.FedFreeze_Modules.local_freezer.local_freezing(cur_T, self.net)
        else: # Offloading training
            for e in range(E):
                logger.debug('Epoch: {:}'.format(E))
                for _, (inputs, targets) in enumerate(trainloader):
                    if update_flag:
                        with torch.no_grad():
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = self.net(inputs)
                            outputs = outputs.to('cpu')
                            # Sending activation
                            msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', self.EcoFed_Modules.compressor.compress(outputs), targets, id]
                            self.communicator.send_msg(self.communicator.sock, msg)

        self.net.to('cpu')
        msg = ['MSG_ROUND_FINISH']
        self.communicator.send_msg(self.communicator.sock, msg)

        self.communicator.recv_msg(self.communicator.sock) # MSG_GLOBAL_ROUND_FINISH
        return time_acv_comm
    
    def upload(self):
        if self.pp == -1:
            tic_aggre_comm = time.time()
            truncated_updated_weights = collections.OrderedDict()
            for k in self.net.state_dict().keys():
                if self.FedFreeze_Modules.global_freezer.frozen_dict[k] == 0:
                    truncated_updated_weights[k] = self.net.state_dict()[k]

            msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', truncated_updated_weights, self.ip]
            self.communicator.send_msg(self.communicator.sock, msg)
            self.communicator.recv_msg(self.communicator.sock) # MSG_TIME_RECORD
            time_aggre_comm = time.time() - tic_aggre_comm
        else:
            time_aggre_comm = 0
        return time_aggre_comm


