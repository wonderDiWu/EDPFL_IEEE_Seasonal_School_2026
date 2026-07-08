# FL Server class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
import threading
import time
from tqdm import tqdm
import copy
import numpy as np

from communicator import *
from .FL_Server import *
import utils

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
class DPFL_Server(FL_Server):
    def __init__(self, ip, server_port, dataset, device, path, K, model, num_classes, client_sampler, shard_indices, pp, e, iters):
        self.pp = pp
        self.e = e
        self.iters = iters
        super(DPFL_Server, self).__init__(ip, server_port, dataset, device, path, K, model, num_classes, client_sampler, shard_indices)

    def initialize(self, R, LR, pretrained_init, pretrained_init_path):
        self.nets = {}
        self.optimizers = {}
        self.criterions = {}
        self.time_ini = {}
            
        for client_ip in self.client_ips:
            ## Weight initilization for each round
            if self.pp == -1:
                if R == 0: # First round initilization
                    if pretrained_init:
                        holistic_pretrain_weights = utils.transfer_weights_holistic(pretrained_init_path, self.uninet.state_dict())
                        self.uninet.load_state_dict(holistic_pretrain_weights)
                        init_cweights = self.uninet.state_dict()
                    else:
                        init_cweights = self.uninet.state_dict()
                else: # Other rounds
                        init_cweights = self.uninet.state_dict()
            else:
                ## Offloading networks
                self.nets[client_ip] = utils.get_partition_model(self.model, self.num_classes, self.device, self.pp, 'Server') # Get partition network
                logger.debug(self.nets[client_ip])
                 
                ## cweight is init weights for client's model 
                cweights = utils.get_partition_model(self.model, self.num_classes, self.device, self.pp, 'Device').state_dict() # Get partition network dict
                self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR, momentum=0.9)
                self.criterions[client_ip] = nn.CrossEntropyLoss()
                
                if R == 0: # First round initilization
                    ## Weight initilization
                    if pretrained_init:
                        holistic_pretrain_weights = utils.transfer_weights_holistic(pretrained_init_path, self.uninet.state_dict())
                        self.uninet.load_state_dict(holistic_pretrain_weights)
                        init_cweights = utils.split_weights(self.uninet.state_dict(), cweights)
                    else:
                        init_cweights = utils.split_weights(self.uninet.state_dict(), cweights)
                    ## pweight is init weights for server's model
                    sweights = utils.split_weights(self.uninet.state_dict(), self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(sweights)
                else: # Other rounds
                    init_cweights = utils.split_weights(self.uninet.state_dict(), cweights)
                    sweights = utils.split_weights(self.uninet.state_dict(), self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(sweights)
                    self.init_cweights = init_cweights
        self.criterion = nn.CrossEntropyLoss() #Used for test
        
        # Weight distribution with multiple threads
        self.threads = {}
        for client_ip in self.client_ips:
            self.threads[client_ip] = threading.Thread(target=self._thread_weights_distribution_, args=(client_ip, init_cweights,))
            logger.debug(str(client_ip) + ' weights distribution start.')
            self.threads[client_ip].start()

        for client_ip in self.client_ips:
            self.threads[client_ip].join()

        for client_ip in self.client_ips:
            logger.debug(str(client_ip) + ' weights distribution finish.')

    def train(self, r, c):
        # Training start
        self.time_grad = {}
        if self.pp == -1:
            ## Device native training
            self.training_no_offloading()

            for client_ip in self.client_ips:
                self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
                logger.debug('MSG_ROUND_FINISH')
        else:
            self.threads = {}
            for client_ip in self.client_ips:
                self.threads[client_ip] = threading.Thread(target=self._thread_training_offloading, args=(client_ip,))
                logger.debug(str(client_ip) + ' offloading training start')
                self.threads[client_ip].start()

            for client_ip in self.client_ips:
                self.threads[client_ip].join()

            for client_ip in self.client_ips:
                self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
                logger.debug('MSG_ROUND_FINISH')
        
        for client_ip in self.client_ips:
            self.communicator.send_msg(self.client_socks[client_ip], ['MSG_GLOBAL_ROUND_FINISH']) #MSG_GLOBAL_ROUND_FINISH

    def _thread_training_offloading(self, client_ip):
        self.time_grad[client_ip] = 0

        for e in range(self.e):
            for i in tqdm(range(self.iters)):
                # Receiving activation
                msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')

                smashed_layers = msg[1]
                labels = msg[2]
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                
                self.optimizers[client_ip].zero_grad()
                with self.lock:
                    outputs = self.nets[client_ip](inputs)		
                loss = self.criterions[client_ip](outputs, targets)
                loss.backward()
                clip_grad_norm_(self.nets[client_ip].parameters(), max_norm=10)
                self.optimizers[client_ip].step()

                # Sending gradient
                msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]  
                self.communicator.send_msg(self.client_socks[client_ip], msg)

    def aggregate(self, N, K, C):
        w_local_list =[]
        self.msgs = []
        self.threads = {}
        for client_ip in self.client_ips:
            self.threads[client_ip] = threading.Thread(target=self._thread_weights_collection_, args=(client_ip,))
            logger.debug(str(client_ip) + ' weights collection start.')
            self.threads[client_ip].start()

        for client_ip in self.client_ips:
            self.threads[client_ip].join()

        for client_ip in self.client_ips:
                logger.debug(str(client_ip) + ' weights collection finish.')
        
        for msg in self.msgs:
            msg[1] = utils.move_state_dict_to_device(msg[1], self.device)
            ip = msg[2]
            if self.pp == -1:  
                w_local = (msg[1], 1 / (K * C))
                w_local_list.append(w_local)
            else:
                w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[ip].state_dict()),1 / (K * C))
                w_local_list.append(w_local)
                    
        if self.agg_model is None: 
            self.agg_model = utils.zero_init(self.aggnet).state_dict()
        self.agg_model = utils.fed_avg(self.agg_model, w_local_list)
        self.agg_count += K
        logger.info('simulation_agg_control: {:}'.format(self.agg_count))
        if self.agg_count > (K * C):
            assert 'Aggregration error!'
        if self.agg_count == (K * C):
            # Finish training of a round
            self.uninet.load_state_dict(self.agg_model)
            self.agg_model = None
            self.agg_count = 0