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

import utils
from communicator import *
from .Client import Client

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FL_Client(Client):
    def __init__(self, server_addr, server_port, ip, index, device, model):
        super(FL_Client, self).__init__()
        self.ip = ip
        self.index = index
        self.port = server_port + index + 1
        if 'cuda' in device:
            if not torch.cuda.is_available():
                assert "No GPU Found!"
        self.device = device
        self.model = model

        logger.debug('Connecting to Server.')
        self.communicator = Socket.Socket()
        self.communicator.sock.bind((self.ip, self.port))
        print(repr(server_addr), type(server_addr))
        print(repr(server_port), type(server_port))
        print(self.communicator.sock)
        print(self.communicator.sock.family)
        self.communicator.sock.connect((server_addr,server_port))
        # Replace the ip with ip:port
        self.ip = str(self.ip)+':'+str(self.port)

        logger.debug('Receiving Client Sampler and Shard Indices..')
        msg = self.communicator.recv_msg(self.communicator.sock)
        self.client_sampler = msg[1]
        self.shard_indices = msg[2]

    def initialize(self, lr, num_classes):
        self.lr = lr
        logger.debug('Building Model.')
        self.net = utils.get_model(self.model, num_classes, self.device)
        logger.debug(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                      momentum=0.9, weight_decay=5e-4)

        logger.debug('Receiving Global Weights..')
        msg = self.communicator.recv_msg(self.communicator.sock)
        self.communicator.send_msg(self.communicator.sock, ['MSG_TIME_RECORD'])
        weights = msg[1]
        self.net.load_state_dict(weights)
        logger.debug('Initialize Finished')

    def train(self, dataloader, E):
        # Training start
        time_acv_comm = 0
        training_loss = 0
        id, trainloader = dataloader[0], dataloader[1]
        self.net.to(self.device)

        self.net.train()
        for e in range(E):
            logger.debug('Epoch: {:}'.format(E))
            for _, (inputs, targets) in enumerate(tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                #training_loss += loss.item()
                loss.backward()
                clip_grad_norm_(self.net.parameters(), max_norm=10)
                self.optimizer.step()

        self.net.to('cpu')
        msg = ['MSG_ROUND_FINISH']
        self.communicator.send_msg(self.communicator.sock, msg)

        self.communicator.recv_msg(self.communicator.sock) # MSG_GLOBAL_ROUND_FINISH
        return time_acv_comm
    
    def upload(self):
        tic_aggre_comm = time.time()
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.state_dict(), self.ip]
        self.communicator.send_msg(self.communicator.sock, msg)
        self.communicator.recv_msg(self.communicator.sock) # MSG_TIME_RECORD
        time_aggre_comm = time.time() - tic_aggre_comm
        return time_aggre_comm

    def time_profile(self, time_acv_comm, time_aggre_comm, time_total_c):
        msg = ['MSG_TIME_PROFILE', self.ip, time_acv_comm, time_aggre_comm, time_total_c]
        self.communicator.send_msg(self.communicator.sock, msg)


