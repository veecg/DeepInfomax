import torch
import numpy as np

import os
import random
from tqdm import tqdm
from math import sqrt
from matplotlib import pyplot as plt
import argparse


from models import *
from data_loader import data_loader, get_encoder_file
from config import BATCH_SIZE

def L1_dist(object1, object2):
    """
    return L1 distance between two objects.
    Two objects must be in the same dimension
    """
    return torch.sum(torch.abs(object1- object2)).item()


def L2_dist(object1, object2):
    """
    return L2 distance between two objects.
    Two objects must be in the same dimension
    """
    return sqrt(torch.sum((object1-object2)**2).item())


def feature_map_pair(enc_path1, enc_path2, enc_checkpoint):
    """
    Taken the trained encoder as input, return a list whose element is a tuple of the image-feature map pair
    """
    # Environment setup
    dataset = 'cifar10'
    
    
    if torch.cuda.is_available:
        batch_size = BATCH_SIZE*torch.cuda.device_count()
        device = torch.device('cuda:0')
    else:
        batch_size = BATCH_SIZE
        device = torch.device('cpu')


    # load trained encoder
    def load_encoder(enc_file,enc_checkpoint):
        encoder = Encoder()
        encoder = nn.DataParallel(encoder).to(device)
        enc_file = get_encoder_file(enc_file,enc_checkpoint)
        encoder.load_state_dict(torch.load(str(enc_file)))
        encoder.eval()
        return encoder
    encoder1 = load_encoder(enc_path1,enc_checkpoint)
    encoder2 = load_encoder(enc_path2,enc_checkpoint)
        

    # load test data
    _, _, test_batches, _ = data_loader(dataset, batch_size)

    # compute latent space for each image in test batches
    img_M1 = [] #store the image, feature map M pairs
    img_M2 = []
    test_batches = tqdm(test_batches)
    with torch.no_grad():
        for batch, labels in test_batches:
            batch = batch.to(device)
            M1, _, _ = encoder1(batch)
            M2, _, _ = encoder2(batch)

            # move back to cpu for faster computation
            images = batch.to(torch.device('cpu')).unbind(0)
            M1 = M1.to(torch.device('cpu')).unbind(0)
            M2 = M2.to(torch.device('cpu')).unbind(0)


            # construct and store image with its feature map as a tuple in the list
            img_M_pair1 = [pair for pair in zip(images, M1)]
            img_M_pair2 = [pair for pair in zip(images, M2)]
            img_M1 += img_M_pair1
            img_M2 += img_M_pair2
            torch.cuda.empty_cache()
    return img_M1, img_M2

    
def display(img_M,N, K):
    fig, subs = plt.subplots(N, K+1, figsize=(5,5))
    for i, sub in enumerate(subs.ravel()):
        sub.imshow(img_M[i][0].permute(1,2,0))
        sub.axis('off')
        if i==0:
            sub.set_title("Query")
    plt.show()
        

def KNN(target_ids,img_M,N,K,dist='l1'):
    """
    Task 2. un-supervised: clustering
    Compute L2 distance on the feature map of the given image and the rest of the images
    display its KNN nearest neighbors.

    Input:
        img_M: the image-feature map pair
        K: return the K nearest neighbors
    """
    nearest_K=[]
    furthest_K=[]
    for img_id in target_ids:
        # Get the image and its feature map
        target_img, target_M = img_M[img_id]
        furthest_K.append(img_M[img_id])
        # Compute L2 distance and sort the list
        if dist=='l1':
            result = sorted(img_M,key=lambda pair:L1_dist(target_M, pair[1]))
        elif dist=='l2':
            result = sorted(img_M,key=lambda pair:L2_dist(target_M, pair[1]))
        else:
            raise ValueError("Undefined distance metric")
        nearest_K += result[0:K+1] #the first one is the target image itself
        furthest_K += result[-K:]


    # Display the results
    display(nearest_K,N,K)
    display(furthest_K,N,K)

def random_list(n,img_M):
    target_ids = []
    for _ in range(n):
        img_id = random.randrange(0,len(img_M))
        target_ids.append(img_id)
    return target_ids
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream task: clustering')
    parser.add_argument('-d', '--dist', default='l1', type=str, help='distance metric')
    parser.add_argument('-n', '--n', default=4, type=int, help='number of sample images')
    parser.add_argument('-k', '--k', default=10, type=int, help='number of nearest neighbors')
    parser.add_argument('-p1', '--enc_path1', type=str, help='folder name of the encoder model to be used under /output')
    parser.add_argument('-p2', '--enc_path2', type=str, help='folder name of the encoder model to be used under /output')
    parser.add_argument('-c', '--enc_checkpoint', type=int, help='checkpoint number to be used. E.g, 500')

    # loading hyper-parameters
    args = parser.parse_args()
    enc_path1 = args.enc_path1
    enc_path2 = args.enc_path2
    enc_checkpoint = args.enc_checkpoint
    dist = args.dist
    k = args.k
    n = args.n


    img_M_globalDIM, img_M_localDIM = feature_map_pair(enc_path1,enc_path2,enc_checkpoint)
    # Randomly generated a list of image ids
    target_ids = random_list(K,img_M_globalDIM)
    # Compute and display its KNN neighbor using GLOBAL only DIM
    KNN(target_ids,img_M_globalDIM, n,k,dist=dist)
    KNN(target_ids,img_M_localDIM,n,k,dist=dist)








