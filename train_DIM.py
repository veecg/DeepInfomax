# Train encoder
import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam

import os
import argparse
import numpy as np
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import json
import argparse

from config import SEED, ALPHA, BETA, GAMMA, LEARNING_RATE, EPOCH_DIM, BATCH_SIZE
from config import HOME_PATH, folder_creation
from data_loader import data_loader
from models import *
from loss_func import JSD_Loss

# fix random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep InfoMax')
    parser.add_argument('-s', '--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-e', '--epoch', default=EPOCH_DIM, type=int)
    parser.add_argument('-l', '--loss', type='jsd', type = str)
    parser.add_argument('-r', '--restart', default=0, type=int, help='Epoch to restart training from. E.g 100')
    parser.add_argument('-p', '--train_path', default='', type=str, help='Folder name to load trained models if restart.')
    parser.add_argument('-a', '--alpha', default=ALPHA, type=int)
    parser.add_argument('-b', '--beta', default=BETA, type=int)
    parser.add_argument('-g', '--gamma', default=GAMMA, type=int)
    parser.add_argument('-lr', '--learning_rate', default=LEARNING_RATE, type=int)



    # loading hyper-parameters
    args = parser.parse_args()
    batch_size = args.batch_size
    total_epoch = args.epoch
    dataset = args.dataset
    loss_fn = args.loss
    restart_epoch = args.restart
    train_path = args.train_path
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    lr = args.learning_rate


    # GPU setup
    if torch.cuda.is_available:
        batch_size = batch_size*torch.cuda.device_count()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    # load data
    train_batches, val_batches, _, num_iter = data_loader(dataset, batch_size)

    # build model architecture: encoder + MI estimator
    encoder = Encoder()
    encoder = nn.DataParallel(encoder).to(device)
    if loss_fn=='jsd':
        loss_fn = JSD_Loss(alpha, beta, gamma)
        loss_fn = nn.DataParallel(loss_fn).to(device)
    else:
        raise NotImplementedError

    # If restart from before: load the parameters
    # Specify paths to store the log and checkpoints
    if restart_epoch!=0:
        TRAIN_PATH = os.path.join(HOME_PATH,'output',train_path)
        LOG_PATH = os.path.join(TRAIN_PATH,'log')
        CHECKPOINT_PATH = os.path.join(TRAIN_PATH,'checkpoint')
        enc_file = os.path.join(CHECKPOINT_PATH, '_encoder_' + str(restart_epoch) + '.wgt')
        loss_file = os.path.join(CHECKPOINT_PATH, 'I_estimator' + str(restart_epoch) + '.wgt')
        print(str(enc_file))
        print(str(loss_file))
        encoder.load_state_dict(torch.load(str(enc_file)))
        loss_fn.load_state_dict(torch.load(str(loss_file)))
        restart_epoch += 1 # Restart from epoch x == start from epoch x+1
    else:
        TRAIN_PATH = folder_creation(HOME_PATH,'train')
        LOG_PATH = folder_creation(TRAIN_PATH,'log')
        CHECKPOINT_PATH = folder_creation(TRAIN_PATH,'checkpoint')

    # Configure Tensorboard
    configure(LOG_PATH)

    # build optimizer, learning rate scheduler
    optim = Adam(list(encoder.parameters())+list(loss_fn.parameters()), lr=lr)


    loss_logger = {"train": [], "valid": []}
    itr = restart_epoch *  num_iter
    for epoch in range(restart_epoch,total_epoch+1):
        train_batches = tqdm(train_batches, total=num_iter)
        train_loss = []
        # start training
        encoder.train()
        loss_fn.train()
        for batch, _ in train_batches:
            batch = batch.to(device)
            optim.zero_grad()
            M, fc, Y   = encoder(batch)
        
            # Calculate loss
            loss = loss_fn(Y, M)
            loss = loss.mean()
            train_loss.append(loss.item())
            # backprop
            loss.backward()
            train_batches.set_description(str(epoch) + ' Loss: ' + str(np.mean(train_loss[-20:])))
            # update parameter
            optim.step()
            # every 10 iteration, log the loss on the tensorboard
            if itr%10==0:
                log_value('train_loss', loss.item(), itr)
                loss_logger['train'].append((loss.item(), itr))
            itr += 1
                

        # For each epoch, check validation loss
        val_loss = []
        encoder.eval()
        loss_fn.eval()
        val_batch = 10
        for _ in range(val_batch):
            batch, _ = next(val_batches)
            batch = batch.to(device)
            M, fc, Y   = encoder(batch)
            loss = loss_fn(Y, M)
            loss = loss.mean()
            val_loss.append(loss.item())
        # Log value
        log_value('val_loss', np.mean(val_loss), epoch)
        loss_logger['valid'].append((np.mean(val_loss), epoch))

        with open(os.path.join(LOG_PATH, "loss_logger.json"), 'w') as f:
            json.dump(loss_logger, f)

        # For every five epochs, save the model weight
        if epoch % 5 == 0:
            enc_file = os.path.join(CHECKPOINT_PATH, '_encoder_' + str(epoch) + '.wgt')
            loss_file = os.path.join(CHECKPOINT_PATH, 'I_estimator' + str(epoch) + '.wgt')
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))

                
