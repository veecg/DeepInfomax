# Data Loader
# Dataset option
#   CIFAR10
#   CIFAR100
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from config import SEED, HOME_PATH


def valid_iter(val_data,batch_size,to_GPU):
    # loop through the validation set infinitely
    data = DataLoader(val_data, batch_size, shuffle=True, drop_last=True, pin_memory=to_GPU)
    while True:
        for x, label in data:
            yield x, label

def get_encoder_file(path,checkpoint):
    return os.path.join(HOME_PATH,'output',path,'checkpoint','_encoder_'+str(checkpoint)+'.wgt')
     

def get_cls_checkpoint(path, checkpoint):
    return os.path.join(
        HOME_PATH,'task_result','classification',path,'checkpoint',
        'classifier'+str(checkpoint)+'.wgt')


def checkpoint_parser(params_file):
    # To remove the "module" in the key name "module.c0.weight" 
    # So that the checkpoint can be properly loaded 
    params_file = torch.load(str(params_file))
    new_params_file = {}
    for key, value in params_file.items():
        # To remove the "module" in the key name "module.c0.weight" 
        keys = key.split('.')[1:]
        keys = ".".join(keys)
        new_params_file[keys] = value
    return new_params_file

def data_loader(dataset, batch_size, val_portion=1):
    """
    Input:
        - val_portion: portion of the validation set; e.g 1 -> 1/10 of the training sets are validation set
    
    Output:  Batch generator that returns elements from dataset batch by batch
        train_data: a batch generator for training data
        val_data:   a batch generator for validation data; inf loop through the validation set
        test_data:  test dataset
        num_iter:   number of iterations per epoch

    """
    # load data
    if dataset == 'cifar10':
#         data = CIFAR10(root='/home/vivian66/projects/def-coama74/vivian66/ecse626/cifar10', download=True, transform=ToTensor())
#         test_data = CIFAR10(root='/home/vivian66/projects/def-coama74/vivian66/ecse626/cifar10', train=False, transform=ToTensor())
        data = CIFAR10(root='data/', download=True, transform=ToTensor())
        test_data = CIFAR10(root='data/', train=False, transform=ToTensor())
    else:
        raise NotImplementedError

    classes = data.classes

    # split training set into training and validation sets
    val_size = len(data) * val_portion/10
    train_size = len(data) - val_size
    print('val_size',val_size)
    print('train_size',train_size)
    train_data, val_data = random_split(data, [int(train_size), int(val_size)], generator=torch.Generator().manual_seed(SEED))
   
    # generate batches
    to_GPU = torch.cuda.is_available()
    train_batches = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, pin_memory=to_GPU)
    val_batches = valid_iter(val_data,batch_size,to_GPU)
    test_batches = DataLoader(test_data, batch_size, shuffle=True, drop_last=True, pin_memory=to_GPU)
    
    
    num_iter = len(train_data)//batch_size

    return train_batches, val_batches, test_batches, num_iter