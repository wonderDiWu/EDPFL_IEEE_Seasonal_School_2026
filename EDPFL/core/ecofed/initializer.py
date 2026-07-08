'''
Pre-training with PyTorch.
code source: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import yaml
import sys
sys.path.append('../')
import time

import utils

class Initializer:
    def __init__(self, ecofed_config):
        self.pretrained_dataset_name = ecofed_config['Initializer']['Dataset']['name']
        self.pretrained_dataset_path = ecofed_config['Initializer']['Dataset']['path']
        self.pretrained_net_path = ecofed_config['Initializer']['Pre-training']['save_path']
        self.model_name = ecofed_config['Initializer']['Pre-training']['model_name']
        self.num_classes = ecofed_config['Initializer']['Dataset']['num_classes']
        self.device = ecofed_config['Initializer']['Pre-training']['device']
        if 'cuda' in self.device:
            if not torch.cuda.is_available():
                assert "No GPU Found!"
        self.pretrained_net_file_name = self.model_name + '_' + self.pretrained_dataset_name + '.pth'

        self.lr = ecofed_config['Initializer']['Pre-training']['LR']
        self.epochs = ecofed_config['Initializer']['Pre-training']['EPOCHS']
    
    def pre_training(self):
        # Concatenate the path and file name
        file_path = os.path.join(self.pretrained_net_path, self.pretrained_net_file_name)

        # Check if the pretrained weights exist
        print(repr(file_path))
        print(file_path)
        if os.path.exists(file_path):
            print("pretrained "+ self.pretrained_net_file_name +" found.")
        else:
            print("Pretraining starting..")
            self.train()
        return file_path
    
    def train(self):
        # Data
        print('==> Preparing data..')
        if self.pretrained_dataset_name == 'Tiny-ImageNet':
            transform_train = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            trainset = datasets.ImageFolder(self.pretrained_dataset_path +'train', transform=transform_train)
            trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=32)
        else:
            raise NotImplementedError("This feature is not yet implemented.")

        best_acc = 0  # best test accuracy

        # Model
        print('==> Building model..')
        net = utils.get_model(self.model_name, self.num_classes, self.device)
        net = net.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        milestones = [int(self.epochs * 0.5), int(self.epochs * 0.8)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        start_time = time.time()
        net.train()
        for epoch in range(self.epochs):
            print('\nEpoch: %d' % epoch)
            train_loss = 0
            correct = 0
            total = 0
            for _, (inputs, targets) in enumerate(tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:
                save_file = os.path.join(self.pretrained_net_path, self.pretrained_net_file_name)
                torch.save(net.state_dict(), save_file)
                best_acc = acc

            scheduler.step()

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Pre-training took {training_time} seconds")
    

def unit_test():
    # Loading config hyper-parametes
    with open('../ecofed_config.yml','r') as f:
        ecofed_config = yaml.load(f, Loader=yaml.FullLoader)
    pretrained_initializer = Initializer(ecofed_config)
    pretrained_initializer.pre_training()

if __name__ == '__main__':
    if unit_test():
        print('Unit test pass!')