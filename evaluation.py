import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from data_loader import get_cls_checkpoint, checkpoint_parser, data_loader
from models import Classifier
from config import BATCH_SIZE
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classifier evaluation')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('-p', '--cls_path', type=str, help='folder name of the classifier model to be used under /task_result/classification. e.g conv1203_2236')
    parser.add_argument('-c', '--cls_checkpoint', type=int, help='checkpoint number to be used. E.g, 200')
    parser.add_argument('-l', '--layer', default='', type=str, help='Specify if the given classifier is using DIM encoder intermidiate layer output')
    

def classification_accuracy(dataset,cls_path, cls_checkpoint, layer, fully_supervised):
    """
        Given a trained classifier, return the classification accuracy on CIFAR10 test data
    """

    args = parser.parse_args()
    dataset = args.dataset
    cls_path = args.cls_path
    cls_checkpoint = args.cls_checkpoint
    layer = args.layer


    # GPU setup
    if torch.cuda.is_available:
        device = torch.device('cuda:0')   
        batch_size = BATCH_SIZE*torch.cuda.device_count()  
    else:
        device = torch.device('cpu') 
        batch_size = BATCH_SIZE
        

    # Load classifier
    cls_file = get_cls_checkpoint(cls_path,cls_checkpoint)
    print("Checkpoint to be loaded:",cls_file)
    cls_file = checkpoint_parser(cls_file)
    fully_supervised = True if layer=='' else False
    classifier = Classifier(eval=True, layer=layer, fully_supervised=fully_supervised)
    classifier.load_state_dict(cls_file)
    classifier = nn.DataParallel(classifier).to(device)

    # Load test data
    _, _, test_batches, _ = data_loader(dataset, batch_size)

    # Accuracy evaluation
    acc = []
    test_batches = tqdm(test_batches)
    for batch, test_label in test_batches:
        batch = batch.to(device)
        y = classifier(batch)
        _, predict_label = y.max(1)
        predict_label = predict_label.to(torch.device('cpu'))
        
        # from tensor to numpy array
        predict_label = predict_label.numpy()
        test_label = test_label.numpy()
        
        batch_acc = predict_label == test_label
        batch_acc = batch_acc.tolist()
        
        
        acc += batch_acc
    
    print(np.mean(acc))

