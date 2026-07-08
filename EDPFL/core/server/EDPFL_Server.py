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
import sys
import random
import collections

from communicator import *
from .FL_Server import *

import utils

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
class EDPFL_Server(FL_Server):
    def __init__(self, ip, server_port, dataset, device, path, K, model, num_classes, client_sampler, shard_indices, e, iters, FedAdapt_Modules, EcoFed_Modules, FedFreeze_Modules):
        self.e = e
        self.iters = iters
        self.FedAdapt_Modules = FedAdapt_Modules
        self.EcoFed_Modules = EcoFed_Modules
        self.FedFreeze_Modules = FedFreeze_Modules
        super(EDPFL_Server, self).__init__(ip, server_port, dataset, device, path, K, model, num_classes, client_sampler, shard_indices)

        # FedAdapt pre_profiling
        self.observations = {}
        for client_ip in self.client_ips:
            msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_OBSERVATION')
            self.observations[client_ip] = msg[1]
        rep_client_ip = self.FedAdapt_Modules.clustering.fit(self.observations)
        self.FedAdapt_Modules.agent.pre_profiling(rep_client_ip)
        
        # initialize global freezer and convergence monitor
        self.FedFreeze_Modules.global_freezer.build(self.uninet)
        self.FedFreeze_Modules.convergence_monitor.build(copy.deepcopy(self.uninet.state_dict()), self.FedFreeze_Modules.global_freezer.layers_dict_with_types)
        self.FedFreeze_Modules.layer_regularizer.build()

    def initialize(self, R, LR, _, __):
        self.nets = {}
        self.optimizers = {}
        self.criterions = {}
        self.time_ini = {}
        self.pps = {}
        self.init_cweights = {}

        # FedAdapt observations
        self.observations = {}

        # Weight distribution with multiple threads
        self.threads = {}
        for client_ip in self.client_ips:

            # FedAdapt adaptive offloading
            msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_OBSERVATION')
            self.observations[client_ip] = msg[1]
            group_id = self.FedAdapt_Modules.clustering.predict(self.observations[client_ip])
            group_action = self.FedAdapt_Modules.agent.action(group_id)
            self.pps[client_ip] = self.FedAdapt_Modules.postprocessor.postprocess(group_action)
            
            if '52001' in client_ip:
                self.pps[client_ip] = 1
            else:
                self.pps[client_ip] = 1

            # Sending partitioning point to devices
            msg = ['MSG_PP', self.pps[client_ip]]
            self.communicator.send_msg(self.client_socks[client_ip], msg)

            ## Weight initilization for each round
            if self.pps[client_ip] == -1:
                if R == 0: # First round initilization
                    holistic_pretrain_weights = utils.transfer_weights_holistic(self.EcoFed_Modules.initializer.pre_training(), self.uninet.state_dict())
                    self.uninet.load_state_dict(holistic_pretrain_weights)
                    self.init_cweights[client_ip] = self.uninet.state_dict()
                else: # Other rounds
                    self.init_cweights[client_ip] = self.uninet.state_dict()
            else:
                ## Offloading networks
                self.nets[client_ip] = utils.get_partition_model(self.model, self.num_classes, self.device, self.pps[client_ip], 'Server') # Get partition network
                logger.debug(self.nets[client_ip])
                
                ## cweight is init weights for client's model 
                cweights = utils.get_partition_model(self.model, self.num_classes, self.device, self.pps[client_ip], 'Device').state_dict() # Get partition network dict
                self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR, momentum=0.9)
                self.criterions[client_ip] = nn.CrossEntropyLoss()
                
                if R == 0: # First round initilization
                    ## Weight initilization
                    holistic_pretrain_weights = utils.transfer_weights_holistic(self.EcoFed_Modules.initializer.pre_training(), self.uninet.state_dict())
                    self.uninet.load_state_dict(holistic_pretrain_weights)
                    self.init_cweights[client_ip] = utils.split_weights(self.uninet.state_dict(), cweights)

                    ## pweight is init weights for server's model
                    sweights = utils.split_weights(self.uninet.state_dict(), self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(sweights)
                else: # Other rounds
                    self.init_cweights[client_ip] = utils.split_weights(self.uninet.state_dict(), cweights)
                    sweights = utils.split_weights(self.uninet.state_dict(), self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(sweights)

            # Current global frozen dict
            global_frozen_dict = self.FedFreeze_Modules.global_freezer.frozen_dict
            logger.debug(global_frozen_dict)

            adaptive_mu_factor = self.FedFreeze_Modules.layer_regularizer.adaptive_mu_factor
            logger.debug('Adaptive mu factor: {:}'.format(adaptive_mu_factor))
            self.threads[client_ip] = threading.Thread(target=self._thread_weights_distribution_, args=(client_ip, self.init_cweights[client_ip], global_frozen_dict, adaptive_mu_factor))
            logger.debug(str(client_ip) + ' weights distribution start.')
            self.threads[client_ip].start()

        for client_ip in self.client_ips:
            self.threads[client_ip].join()

        for client_ip in self.client_ips:
            logger.debug(str(client_ip) + ' weights distribution finish.')

        # Test dataset loss function
        self.criterion = nn.CrossEntropyLoss()     

    def train(self, r, c):
        # Training start
        self.time_grad = {}
        self.threads = {}
        self.random_ids = {}
        self.buffer_available = {}

        # Monitoring buffer size and update flag
        logger.info('Buffer size: # pp x id {:}'.format(self.EcoFed_Modules.replay_buffer.get_buffer_size()))
        logger.info('Update flag: {:}'.format(self.EcoFed_Modules.activation_switch.ask()))
        if not self.EcoFed_Modules.activation_switch.ask():
            for client_ip, pp in self.pps.items():
                if pp != -1: # Exclude the device native training
                    if pp in self.EcoFed_Modules.replay_buffer.buffer.keys():
                        avaliable_ids = list(self.EcoFed_Modules.replay_buffer.buffer[pp].keys())
                        self.buffer_available[client_ip] = True
                        self.random_ids[client_ip] = random.choice(avaliable_ids)
                    else:
                        self.buffer_available[client_ip] = False

        
        for client_ip in self.client_ips:  
            if self.pps[client_ip] == -1:
                ## Device native training
                self.threads[client_ip] = threading.Thread(target=self._training_no_offloading, args=(client_ip,))
                logger.debug(str(client_ip) + ' no offloading training start')
                self.threads[client_ip].start()
            else:
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

    def _thread_weights_distribution_(self, client_ip, init_cweights, global_frozen_dict, adaptive_mu_factor):
        tic_ini = time.time()
        msg = ['MSG_GLOBAL_FROZEN_DICT_SERVER_TO_CLIENT', global_frozen_dict, adaptive_mu_factor]
        self.communicator.send_msg(self.client_socks[client_ip], msg)

        msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_LOADING_FLAG')
        local_loading_flag = msg[1]
        if local_loading_flag:
            truncated_init_cweights = collections.OrderedDict()
            for k in init_cweights.keys():
                if global_frozen_dict[k] == 0:
                    truncated_init_cweights[k] = init_cweights[k]
            msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', truncated_init_cweights]
            self.communicator.send_msg(self.client_socks[client_ip], msg)
            self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_TIME_RECORD')
            self.time_ini[client_ip] = time.time() - tic_ini
        else:
            msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', init_cweights]
            self.communicator.send_msg(self.client_socks[client_ip], msg)
            self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_TIME_RECORD')
            self.time_ini[client_ip] = time.time() - tic_ini

    def _thread_weights_collection_(self, client_ip):
        if self.pps[client_ip] == -1:
            msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            self.communicator.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD
        else:
            msg = ['MSG_LOCAL_WEIGHTS_FROM_INIT_WEIGHTS', self.init_cweights[client_ip], client_ip]
        self.msgs.append(msg)

    def _training_no_offloading(self, client_ip): # Thread function for device-native training
        self.time_grad[client_ip] = 0

    def _thread_training_offloading(self, client_ip):
        self.time_grad[client_ip] = 0
        
        # Global freezing
        self.FedFreeze_Modules.global_freezer.freeze_layers_by_names(self.nets[client_ip])

        # Partitioning point
        pp = self.pps[client_ip]

        # Set activation switch flag and send to devices
        if self.EcoFed_Modules.activation_switch.ask() or not self.buffer_available[client_ip]:
            update_flag= True
        else:
            update_flag = False
            random_id = self.random_ids[client_ip]
        msg = ['Update Flag', update_flag]
        self.communicator.send_msg(self.client_socks[client_ip], msg)

        for e in range(self.e):
            for i in tqdm(range(self.iters)):
                if update_flag:
                    # Receiving activation
                    msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                    self.EcoFed_Modules.replay_buffer.update(pp, msg[3], i, msg[1], msg[2])
                    smashed_layers = self.EcoFed_Modules.replay_buffer.buffer[pp][msg[3]][i].activations
                    labels = self.EcoFed_Modules.replay_buffer.buffer[pp][msg[3]][i].labels
                else:
                    smashed_layers = self.EcoFed_Modules.replay_buffer.buffer[pp][random_id][i].activations
                    labels = self.EcoFed_Modules.replay_buffer.buffer[pp][random_id][i].labels

                smashed_layers = self.EcoFed_Modules.compressor.decompress(smashed_layers)
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                
                self.optimizers[client_ip].zero_grad()
                with self.lock:
                    outputs = self.nets[client_ip](inputs)		
                loss = self.criterions[client_ip](outputs, targets)
                loss.backward()
                clip_grad_norm_(self.nets[client_ip].parameters(), max_norm=10)
                self.optimizers[client_ip].step()

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
            truncated_updated_weights = msg[1]
            ip = msg[2]
            if self.pps[ip] == -1:
                self.init_cweights[ip] = utils.move_state_dict_to_device(self.init_cweights[ip], self.device)
                weights = utils.concat_frozen_updated_weights(self.uninet.state_dict(), self.init_cweights[ip], truncated_updated_weights)
                w_local = (weights, 1 / (K * C))
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

    def global_freezing(self):
        norm_eps, rate_eps = self.FedFreeze_Modules.convergence_monitor.check_layer_convergence(copy.deepcopy(self.uninet.to(self.device).state_dict()))
        self.FedFreeze_Modules.global_freezer.global_freezing(norm_eps, rate_eps)

        # Updating local layer regularizer mu factor
        self.FedFreeze_Modules.layer_regularizer.adaptive_mu_factor = sum(self.FedFreeze_Modules.convergence_monitor.norm_eps.values()) / len(self.FedFreeze_Modules.convergence_monitor.norm_eps)