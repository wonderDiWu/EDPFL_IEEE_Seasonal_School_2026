import yaml
import argparse

import time
import pickle
import sys
import os
import numpy as np
import random
import copy
import torch

import utils
from server import *
from client_sampler import *
from fedadapt import FedAdapt
from ecofed import EcoFed
from fedfreeze import FedFreeze

# Dibugging
# import ipdb

# Log and Visulization
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--testbed', type= str, dest= 'testbed', required= True)
    parser.add_argument('--dataset', type= str, dest= 'dataset', required= True)
    parser.add_argument('--model', type= str, dest= 'model', required= True)
    parser.add_argument('--mode', type= str, dest= 'mode', required= True)
    args = parser.parse_args()

    TESTBED = args.testbed
    DATASET = args.dataset
    MODEL = args.model # Model name
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
    ITERS = int((N / ((K * C) / ALPHA * B))) # Local iterations

    # DPFL configrations
    PP = config['DPFL']['PP'] # Partition point

    # System configrations
    DEVICE = config['DEVICE']

    # Visulization configrations
    WANDB = config['WANDB']
    if WANDB:
        import wandb
        #os.environ['WANDB_DISABLED'] = 'true'
        wandb.init(project='EDPFL', entity='datawonder8')

    # Random Seed
    SEED = config['FL']['SEED'] 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_deterministic_debug_mode('default')
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Generating shards indices for local data
    shard_indices = random.sample(range(NUM_SHARDS), NUM_SHARDS)

    # Sampling clients for FL rounds
    client_sampler = Random_Sampler.Random_Sampler(K, C, ALPHA, R, SEED)
    client_sampler.get_samplers()

    # Initialization
    logger.debug('Building Server.')
    if MODE == 'FL':
        server = FL_Server.FL_Server(SERVER_ADDR, SERVER_PORT, DATASET, DEVICE, PATH, K, MODEL, NUM_CLASSES, client_sampler, shard_indices)
    if MODE == 'DPFL':
        server = DPFL_Server.DPFL_Server(SERVER_ADDR, SERVER_PORT, DATASET, DEVICE, PATH, K, MODEL, NUM_CLASSES, client_sampler, shard_indices, PP, E, ITERS)
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

        server = EDPFL_Server.EDPFL_Server(SERVER_ADDR, SERVER_PORT, DATASET, DEVICE, PATH, K, MODEL, NUM_CLASSES, client_sampler, shard_indices, E, ITERS, FedAdapt_Modules, EcoFed_Modules, FedFreeze_Modules)

    # Record experiment config to wandb
    if WANDB:
        wandb_config = wandb.config
        wandb_config['TESTBED'] = TESTBED
        wandb_config['DATASET'] = '_'.join([str(item) for item in [str(DATASET),NON_IID]])
        wandb_config['MODEL'] = MODEL
        wandb_config['TRAINING_PARAMETERS'] = '_'.join([str(item) for item in [K, C, ALPHA, NON_IID, R, E, B, LR, LR_SCHEDULE, LR_FACTOR, PRETRAINED_INIT, PRETRAINED_INIT_PATH, SEED]])
        wandb_config['MODE'] = MODE

    # Training start
    metric = {} # Recording the resules
    accumulated_time = 0
    for r in range(R):
        if r == 0:
            test_acc, test_loss = server.test()
            metric['Global Accuracy'] = []
            metric['Global Accuracy'].append(test_acc)
            metric['Global loss'] = []
            metric['Global loss'].append(test_loss)
            metric['Wall-clock Time'] = []
            metric['Wall-clock Time'].append(0)
            metric['Communication Cost'] = []
            metric['Communication Cost'].append(0)
            if MODE == 'EDPFL': 
                metric['EPS'] = []
                metric['EPS'].append(server.FedFreeze_Modules.convergence_monitor.eps)
            logger.info('Global loss: {:}, Global Acc: {:}'.format(test_loss, test_acc))
            if WANDB:
                if MODE == 'EDPFL':
                    wandb_res = copy.deepcopy(server.FedFreeze_Modules.convergence_monitor.rate_eps)
                else:
                    wandb_res = {}
                wandb_res['Gobal loss'] = test_loss
                wandb_res['Global Acc'] = test_acc
                wandb_res['Wall-clock Time'] = 0
                wandb_res['Communication Cost'] = 0
                wandb.log(wandb_res)
    
        tic_total = time.time()
        logger.info('='*100 + '=>')
        logger.info('Round {:} Start'.format(r))
        logger.info('Current Client IDs: ')
        logger.info(server.client_sampler.indices_samples[r])

        time_round = []
        for c in range(C):
            tic_total = time.time()
            logger.info('Cluster {:} Start'.format(c))
            adjust_lr = utils.adjust_lr(r, LR, LR_FACTOR, LR_SCHEDULE)
            logger.debug('LR: {}'.format(adjust_lr))
            server.initialize(r, adjust_lr, PRETRAINED_INIT, PRETRAINED_INIT_PATH)
            logger.debug('Initialization Finish')

            server.train(r, c)

            comm_upload_c = server.aggregate(N, K ,C)
            logger.debug('Cluster Finish')
            time_total_c = time.time() - tic_total # Training time for each simulated cluster
            time_round.append(time_total_c)
            
        accumulated_time += max(time_round)
        accumulated_comm = server.communicator.comm_cost

        if MODE == 'EDPFL':
            # Counting the training round for EcoFed
            server.EcoFed_Modules.activation_switch.round_count += 1

            # FedFreeze global freezing
            server.global_freezing()
            logger.info(server.FedFreeze_Modules.global_freezer.frozen_dict)

        # Global test and local test
        test_acc, test_loss = server.test()
        metric['Global Accuracy'].append(test_acc)
        metric['Global loss'].append(test_loss)
        metric['Wall-clock Time'].append(accumulated_time)
        metric['Communication Cost'].append(accumulated_comm)
        if MODE == 'EDPFL':
            metric['EPS'].append(server.FedFreeze_Modules.convergence_monitor.eps)
        logger.info('Global loss: {:}, Global Acc: {:}'.format(test_loss, test_acc))
        if WANDB:
                if MODE == 'EDPFL':
                    wandb_res = copy.deepcopy(server.FedFreeze_Modules.convergence_monitor.rate_eps)
                else:
                    wandb_res = {}
                wandb_res['Gobal loss'] = test_loss
                wandb_res['Global Acc'] = test_acc
                wandb_res['Wall-clock Time'] = accumulated_time
                wandb_res['Communication Cost'] = accumulated_comm
                wandb.log(wandb_res)
        logger.info('Round Finish')
        
        '''
        if NON_IID:
            with open('../../results/'+DATASET+'/'+MODEL+'/'+'res_'+MODE+'_'+str(K * C)+
            '_'+'Non_IID_' + str(PRETRAINED_INIT) +'_'+ str(ALPHA) +'_'+ str(SEED)+
            '.pkl','wb') as f:
                    pickle.dump(metric,f)
        else:
            with open('../../results/'+DATASET+'/'+MODEL+'/'+'res_'+MODE+'_'+str(K * C)+
            '_'+'IID_'+ str(PRETRAINED_INIT) +'_'+ str(ALPHA) +'_'+ str(SEED)+
            '.pkl','wb') as f:
                    pickle.dump(metric,f)
        '''

if __name__ == "__main__":
    main()