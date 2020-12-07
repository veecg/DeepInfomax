# Downstream tasks:
#   1. supervised: image classification
#       1.1. fully supervised: raw input -> raw encoder -> classifier
#       1.2  use trained ncoder: raw input -> trained encoder -> classifier    


import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam

import os
import numpy as np
from tqdm import tqdm
from tensorboard_logger import configure, log_value

from config import SEED, LEARNING_RATE, HOME_PATH, EPOCH_CLS, BATCH_SIZE
from config import folder_creation
from models import Classifier
from data_loader import data_loader, get_encoder_file, get_restart_checkpoint
import argparse



    
def classification_task(enc_checkpoint_path, checkpoint, task=1 ):
    encoder_file = get_encoder_file(enc_checkpoint_path,checkpoint)
    if task ==1: 
        #fully supervised
        classifier = Classifier(layer=None, fully_supervised=True)
    elif task ==2: 
        #supervised using conv from trained encoder
        classifier = Classifier(layer='conv',params_file=encoder_file)
    elif task ==3:
        #supervised using fc from trained encoder
        classifier = Classifier(layer='fc',params_file=encoder_file)
    elif task ==4:
        #supervised using Y from trained encoder
        classifier = Classifier(layer='Y',params_file=encoder_file)
    else:
        raise ValueError('[!] Invalid classification task number.')
    return classifier

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream task: image classification')
    parser.add_argument('-t', '--task', type=int, help='1: fully supervise; 2:conv; 3: fc; 4:Y')
    parser.add_argument('-p', '--enc_path', type=str, help='folder name of the encoder model to be used under /output')
    parser.add_argument('-c', '--enc_checkpoint', type=int, help='checkpoint number to be used. E.g, 500')
    parser.add_argument('-e', '--epoch', default=EPOCH_CLS, type=int, help='number of epochs to train classifier')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE*2, type=int)
    parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('-r', '--restart', default=0, type=int, help='Epoch to restart training from. E.g 100')
    parser.add_argument('-p', '--train_path', default='', type=str, help='Folder name to load trained models if restart.')
    parser.add_argument('-lr', '--learning_rate', default=LEARNING_RATE, type=int)

    # training environment setup
    args = parser.parse_args()
    task = args.task
    enc_path = args.enc_path
    enc_checkpoint = args.enc_checkpoint
    total_epoch = args.epoch
    batch_size = args.batch_size
    dataset = args.dataset
    restart_checkpoint = args.restart
    restart_folder = args.train_path
    lr = args.learning_rate
    
    # GPU setup
    if torch.cuda.is_available:
        batch_size = batch_size*torch.cuda.device_count()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
   
    

    # model setup
    classifier = classification_task(enc_path,enc_checkpoint,task)
    classifier = nn.DataParallel(classifier).to(device)
    
    # if restart
    if restart_checkpoint!=0:
        cls_file = get_restart_checkpoint(restart_folder,restart_checkpoint)
        classifier.load_state_dict(torch.load(str(cls_file)))
        RESULT_PATH = os.path.join(HOME_PATH,'task_result','classification',restart_folder)
        LOG_PATH = os.path.join(RESULT_PATH,'log')
        CHECKPOINT_PATH = os.path.join(RESULT_PATH,'checkpoint')
        
    else:
        RESULT_PATH = folder_creation(HOME_PATH,'task'+str(task))
        LOG_PATH = folder_creation(RESULT_PATH,'log')
        CHECKPOINT_PATH = folder_creation(RESULT_PATH,'checkpoint')
    
    
    optim = Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Configure Tensorboard
    try:
        configure(LOG_PATH)
    except ValueError:
        print("Tensorboard already configured at", LOG_PATH)
        
    # load data
    train_batches, val_batches, test_batches, num_iter = data_loader(dataset, batch_size)
    
    itr = restart_checkpoint *  num_iter
    for epoch in range(restart_checkpoint,total_epoch+1):
        train_batches = tqdm(train_batches, total=num_iter)
        train_loss = []

        classifier.train()
        for batch, true_labels in train_batches:
            batch = batch.to(device)
            true_labels = true_labels.to(device)

            optim.zero_grad()

            predict_labels = classifier(batch)
            loss = loss_fn(predict_labels, true_labels)

            train_loss.append(loss.item())
            train_batches.set_description(str(epoch)+ ' Train Loss'+ str(np.mean(train_loss[-20:])))

            loss.backward()
            optim.step()

            if itr%10==0:
                log_value('train_loss', loss.item(), itr)
            itr += 1

        # For each epoch, check validation loss
        val_loss = []
        classifier.eval()
        val_batch = 5 # number of batches used for each validation test
        for _ in range(val_batch):
            batch, val_labels = next(val_batches)
            batch = batch.to(device)
            val_labels = val_labels.to(device,dtype=torch.long)
            
            predict_labels  = classifier(batch)
            loss = loss_fn(predict_labels,val_labels)
            loss = loss.mean()
            val_loss.append(loss.item())
        # Log value
        log_value('val_loss', np.mean(val_loss), epoch)
        


        # For every five epochs, save the checkpoint
        if epoch % 5 == 0:
            classifier_file = os.path.join(CHECKPOINT_PATH, 'classifier' + str(epoch) + '.wgt')
            torch.save(classifier.state_dict(), str(classifier_file))

    
    
        
        
       
    



