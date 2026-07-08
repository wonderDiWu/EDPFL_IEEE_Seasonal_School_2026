# FL Server class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import time
from tqdm import tqdm
import copy
import numpy as np

from communicator import *
from .Server import Server
import utils

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
class FL_Server(Server):
    def __init__(self, ip, server_port, dataset, device, path, K, model, num_classes, client_sampler, shard_indices):
        super(FL_Server, self).__init__()
        import os

        print(os.getpid(), "bind")
        self.ip = ip
        self.port = server_port
        self.communicator = Socket.Socket()
        self.communicator.sock.bind((self.ip, self.port))
        self.communicator.sock.listen(5)

        self.client_ips = set()
        self.client_socks = {}
        # For PI testbed
        PORT2IP = {'52001':'192.168.137.104', '52002':'192.168.137.168'}
        # Waiting connection from K devices
        while len(self.client_socks) < K:
            self.communicator.sock.listen(5)
            logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.communicator.sock.accept()
            logger.info('Got connection from ' + str(ip)+':'+str(port))
            logger.info(client_sock)
            ip = PORT2IP[str(port)]
            self.client_ips.add(str(ip)+':'+str(port))
            self.client_socks[str(ip)+':'+str(port)] = client_sock
        
        if 'cuda' in device:
            if not torch.cuda.is_available():
                assert "No GPU Found!"
        self.device = device
        
        # Global model
        self.model = model
        self.num_classes = num_classes
        self.uninet = utils.get_model(self.model, self.num_classes, self.device)
        logger.info(self.uninet)

        # Aggregation
        self.aggnet = utils.get_model(self.model, self.num_classes, self.device)
        self.agg_model = None # agg_model for simulation
        self.agg_count = 0 # number of aggregration times

        # Test sets
        if dataset == 'FMNIST':
            self.transform_test = transforms.Compose([transforms.Pad(padding=(2, 2), fill=0, padding_mode='constant'), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.testset = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=self.transform_test)
        if dataset == 'CIFAR10':
            self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            self.testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=self.transform_test)
        if dataset == 'CIFAR100':
            self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
            self.testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
        self.criterion = nn.CrossEntropyLoss() #Used for test in simulation 

        # Samplers and shard_indices
        self.client_sampler = client_sampler
        self.shard_indices = shard_indices
        for client_ip in self.client_ips:
            msg = ['MSG_CLIENT_SAMPLER_SERVER_TO_CLIENT', self.client_sampler, self.shard_indices]
            self.communicator.send_msg(self.client_socks[client_ip], msg)
        
          # Locking for multiple GPUs
        self.lock = threading.Lock()

    def initialize(self, R, LR, pretrained_init, pretrained_init_path):
        self.nets = {}
        self.optimizers = {}
        self.criterions = {}
        self.time_ini = {}
            
        for client_ip in self.client_ips:
            ## Weight initilization for each round
            if R == 0: # First round initilization
                if pretrained_init:
                    holistic_pretrain_weights = utils.transfer_weights_holistic(pretrained_init_path, self.uninet.state_dict())
                    self.uninet.load_state_dict(holistic_pretrain_weights)
                    init_cweights = self.uninet.state_dict()
                else:
                    init_cweights = self.uninet.state_dict()
            else: # Other rounds
                    init_cweights = self.uninet.state_dict()
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
        ## Device native training
        self.training_no_offloading()
        for client_ip in self.client_ips:
            self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
            logger.debug('MSG_ROUND_FINISH')
        
        for client_ip in self.client_ips:
            self.communicator.send_msg(self.client_socks[client_ip], ['MSG_GLOBAL_ROUND_FINISH']) #MSG_GLOBAL_ROUND_FINISH

    def training_no_offloading(self):
        for client_ip in self.client_ips:
            self.time_grad[client_ip] = 0

    def _thread_weights_distribution_(self, client_ip, init_cweights):
        tic_ini = time.time()
        msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', init_cweights]
        self.communicator.send_msg(self.client_socks[client_ip], msg)
        self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_TIME_RECORD')
        self.time_ini[client_ip] = time.time() - tic_ini

    def _thread_weights_collection_(self, client_ip):
        msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
        self.communicator.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD
        self.msgs.append(msg)

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
            w_local = (msg[1], 1 / (K * C))
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
    
    def test(self):
        self.uninet.eval()
        self.uninet.to(self.device)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.uninet(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        avg_loss = test_loss/total
        self.uninet.to('cpu')

        ## Save checkpoint.
        #torch.save(self.uninet.state_dict(), config.home+'ActionFed/trained_models/'+config.dataset_name+'_'+config.model_name+'_'+config.train_mode+'.pth')
        return acc, avg_loss

    def time_profile(self, time_total_s):
        rec = {}
        for client_ip in self.client_ips:
            msg = self.communicator.recv_msg(self.client_socks[client_ip], 'MSG_TIME_PROFILE')
            ip, time_acv_comm, time_aggre_comm, time_total_c = msg[1], msg[2], msg[3], msg[4] 
            logger.debug('IP: {:}, Total_Client: {:}, Total_Server: {:}'.format(client_ip, time_total_c, time_total_s))
            total_time = max(time_total_c, time_total_s)
            comm_time = self.time_ini[client_ip] + time_acv_comm + self.time_grad[client_ip] + time_aggre_comm
            comp_time = total_time - comm_time
            logger.debug('Init_Comm.: {:}, Act_Comm.: {:}, Grad_Comm.: {:}, Aggr_Comm.: {:}'.format(self.time_ini[client_ip], time_acv_comm, self.time_grad[client_ip], time_aggre_comm))
            logger.debug('Communication: {:}, Computation: {:}, Total: {:}'.format(comm_time, comp_time, total_time))
            
            ## Save the time record
            rec[client_ip] = [client_ip, time_acv_comm, time_aggre_comm, time_total_c, time_total_s, comm_time, comp_time, total_time]
        return rec
