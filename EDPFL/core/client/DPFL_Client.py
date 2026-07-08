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
from .FL_Client import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DPFL_Client(FL_Client):
    def __init__(self, server_addr, server_port, ip, index, device, model, pp):
        self.pp = pp
        super(DPFL_Client, self).__init__(server_addr, server_port, ip, index, device, model)


    def initialize(self, lr, num_classes):
        self.lr = lr
        logger.debug('Building Model.')
        self.net = utils.get_partition_model(self.model, num_classes, self.device, self.pp, 'Device') # Get partition network
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
        if self.pp == -1: # No offloading training
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
        else: # Offloading training
            for e in range(E):
                logger.debug('Epoch: {:}'.format(E))
                for _, (inputs, targets) in enumerate(tqdm(trainloader)):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)

                    # Sending activation
                    msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs, targets] #normal
                    self.communicator.send_msg(self.communicator.sock, msg)

                    # Recieving gradient
                    msg = self.communicator.recv_msg(self.communicator.sock)
                    gradients = msg[1].to(self.device)

                    outputs.backward(gradients)
                    clip_grad_norm_(self.net.parameters(), max_norm=10)
                    self.optimizer.step()

        self.net.to('cpu')
        msg = ['MSG_ROUND_FINISH']
        self.communicator.send_msg(self.communicator.sock, msg)

        self.communicator.recv_msg(self.communicator.sock) # MSG_GLOBAL_ROUND_FINISH
        return time_acv_comm


