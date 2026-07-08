import yaml
import argparse

import socket
import time
import multiprocessing
import distutils
import os
import numpy as np
import random
import torch

import sys
import utils
from client import *
from client_sampler import *
from data_generator import *
from fedadapt import FedAdapt
from ecofed import EcoFed
from fedfreeze import FedFreeze

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()   
parser.add_argument('--testbed', type= str, dest= 'testbed', required= True)
parser.add_argument('--dataset', type= str, dest= 'dataset', required= True)
parser.add_argument('--model', type= str, dest= 'model', required= True)
parser.add_argument('--ip', type= str, dest= 'ip', required= True)
parser.add_argument('--index', type= int, dest= 'index', required= True)
parser.add_argument('--mode', type= str, dest= 'mode', required= True)
args = parser.parse_args()

TESTBED = args.testbed
DATASET = args.dataset
MODEL = args.model # Model name
IP = args.ip
INDEX = args.index
MODE = args.mode # Trainng mode

# Loading config hyper-parametes
with open('config.yml','r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Network configration
SERVER_ADDR = config['Network'][TESTBED]['SERVER_ADDR']
SERVER_PORT = config['Network'][TESTBED]['SERVER_PORT']

# Dataset configration
PATH = config['Dataset'][DATASET]['path']
N = config['Dataset'][DATASET]['N'] # Dataset size
IMG_SIZE = config['Dataset'][DATASET]['img_size'] # Image size
NUM_CLASSES = config['Dataset'][DATASET]['num_classes'] #Number of classes

# FL configration
K = config['FL']['K'] # Number of devices
C = config['FL']['C'] # Number of simulation clusters
ALPHA = config['FL']['ALPHA'] # Sampling ratio in each round
NON_IID = config['FL']['NON_IID'] # Non I.I.D.
NUM_SHARDS = config['FL']['NUM_SHARDS'] # Number of shards
R = config['FL']['R'] # Number of rounds
E = config['FL']['E'] # Number of local epochs
B = config['FL']['B'] # Batchsize
LR = config['FL']['LR'] # Learning rate
LR_FACTOR = config['FL']['LR_FACTOR'] # Learning rate decay factor
LR_SCHEDULE = config['FL']['LR_SCHEDULE'] # Learning rate schedule
PRETRAINED_INIT = config['FL']['PRETRAINED_INIT'] # Pretrained initialization
PRETRAINED_INIT_PATH = config['FL']['PRETRAINED_INIT_PATH'] # Pretrained initialization file location
SEED = config['FL']['SEED'] # Random seed

# DPFL configrations
PP = config['DPFL']['PP'] # Partition point

# System configrations
DEVICE = config['DEVICE']

# Random Seed
SEED = config['FL']['SEED'] 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
#torch.set_deterministic_debug_mode('default')
os.environ['PYTHONHASHSEED'] = str(SEED)

# Building a client instance
logger.debug('Building Client.')
if MODE == 'FL':
    client = FL_Client.FL_Client(SERVER_ADDR, SERVER_PORT, IP, INDEX, DEVICE, MODEL)
if MODE == 'DPFL':
    client = DPFL_Client.DPFL_Client(SERVER_ADDR, SERVER_PORT, IP, INDEX, DEVICE, MODEL, PP)
if MODE == 'EDPFL':
    # Initializing FedAdapt modules
    with open('fedadapt_config.yml','r') as f:
        fedadapt_config = yaml.load(f, Loader=yaml.FullLoader)
    FedAdapt_Modules = FedAdapt.FedAdapt(fedadapt_config)

    # Initializing EcoFed modules
    with open('ecofed_config.yml','r') as f:
        ecofed_config = yaml.load(f, Loader=yaml.FullLoader)
    EcoFed_Modules = EcoFed.EcoFed(ecofed_config)

    # Initializing FedFreeze modules
    with open('fedfreeze_config.yml','r') as f:
        ecofed_config = yaml.load(f, Loader=yaml.FullLoader)
    FedFreeze_Modules = FedFreeze.FedFreeze(ecofed_config)
    
    client = EDPFL_Client.EDPFL_Client(SERVER_ADDR, SERVER_PORT, IP, INDEX, DEVICE, MODEL, PP, NUM_CLASSES, FedAdapt_Modules, EcoFed_Modules, FedFreeze_Modules)

cpu_count = 0
num_clients = int((K * C) / ALPHA)
if NON_IID:
    dataloader_generator = Non_IID_Generator.Non_IID_Generator(cpu_count, DATASET, num_clients, NUM_SHARDS, client.shard_indices, SEED, PATH, B)
else:
    dataloader_generator = IID_Generator.IID_Generator(cpu_count, DATASET, num_clients, NUM_SHARDS, client.shard_indices, SEED, PATH, B)

# Training start
for r in range(R):
    # Client IDs in the current round 
    clinet_ids = client.client_sampler.indices_samples[r]
    for c in range(C):
        logger.debug('Preparing Data.')
        simulation_index = INDEX + c * K
        client_id = clinet_ids[simulation_index]
        logger.debug('Current Client ID {:}'.format(client_id))
        dataloader = dataloader_generator.get_local_dataloader(client_id)
  
        tic_total = time.time()
        logger.debug('====================================>')
        logger.debug('ROUND: {} START'.format(r))
        logger.debug('Cluster: {} START'.format(c))
        adjust_lr = utils.adjust_lr(r, LR, LR_FACTOR, LR_SCHEDULE)
        logger.debug('LR: {}'.format(adjust_lr))
        client.initialize(adjust_lr, NUM_CLASSES)
        logger.debug('==> Initialization Finish')
        
        time_acv_comm = client.train(dataloader, E)

        logger.debug('==> Waiting for aggregration')
        time_aggre_comm = client.upload()
        
        time_total_c = time.time() - tic_total
        #client.time_profile(time_acv_comm, time_aggre_comm, time_total_c)
        #logger.info('Current communication cost: ' + str(client.communicator.comm_cost))
    
    logger.debug('ROUND: {} END'.format(r))
    #logger.debug('Total communication cost: ' + str(client.communicator.comm_cost))