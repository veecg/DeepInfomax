import torch
import torch.nn as nn
import torch.nn.functional as F

import os


# Components of DIM
class Encoder(nn.Module):
    """
    For CIFAR10, CIFAR100;
    Representation generator

    Input:
        x: input image batch; Bx3x32x32

    Output:
        M: MxM feature map; Bx512xMxM
        Y: global feature vector; Bx64
    """
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)

        self.l0 = nn.Linear(512*20*20, 512)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256,64)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)
        self.b4 = nn.BatchNorm2d(512)
        self.b5 = nn.BatchNorm2d(256)
        

    def forward(self,x):
        batch_size = x.shape[0]
        h = F.relu(self.c0(x))
        h = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(h)))
        M = F.relu(self.b3(self.c3(h))) 
        # h = F.relu(self.b4(self.l0(M.view(batch_size,-1))))
        # fc = F.relu(self.b5(self.l1(h)))
        h = F.relu(self.l0(M.view(batch_size,-1)))
        fc = F.relu(self.l1(h))
        Y = self.l2(fc)

        return M, fc, Y


class GlobalDIM(nn.Module):
    """
    p16: Discriminator for Global Objective

    Input:
        M: MxM feature map
        Y: Global feature vector
    Output:
        score: a scalar to indicate relevance of the given pairs
        
    """
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(512*20*20+64,512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512,1)

    def forward(self, Y, M):
        batch_size = Y.shape[0]
        h = M.view(batch_size, -1)
        h = torch.cat((Y,h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        score = self.l2(h)

        return score



class LocalDIM(nn.Module):
    """
    p17: LocalDIM: Concat-and-convolve structure.
    1x1 Conv Discriminator for Global Objective.

    Input:
        M: MxM feature map
        Y: Global feature vector
    Output:
        scores: a MXM score map 
    """
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(512+64, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)
    
    def forward(self, Y, M):
        # Concat
        m = M.shape[-1]
        Y = Y.unsqueeze(-1).unsqueeze(-1) # BxNx1x1
        Y = Y.expand(-1, -1, m, m)
        h = torch.cat((M,Y),dim=1)
        # Convolve
        h = F.relu(self.c0(h))
        h = F.relu(self.c1(h))
        scores = self.c2(h)

        return scores


class PriorDiscriminator(nn.Module):
    """
    Prior matching. 
    Input: 
        Y: global feature vector; Bx64
    
    Output:
        output: discriminator score        
    """
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200,1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        output = torch.sigmoid(self.l2(h))

        return output



# Classifiers for representation evaluation:

# Non-Linear classifier
class Classifier(nn.Module):
    def __init__(self, layer='Y', fully_supervised = False, params_file=None, eval=False):
        """
        Classifier structure option:
            1. fully_supervised=True: encoder + single layer NN
            2. fully_supervised=False: 
            2.1 layer=Y:   Y from the fixed encoder + single layer NN
            2.2 layer=fc:   output from the last two fc layer in the fixed encoder + fc + single layer NN
            2.3 layer=conv:   conv from the fixed encoder + 3*fc + single layer NN
            
            
        Input:
            layer: the encoder layer to be used 
            fully_supervised: if false, the DIM trained encoder will be used
            >> IF fully_supervised==False:
                params_file: the encoder checkpoint file, to load the trained encoder
            eval: if false, return a new classifier to be trained
                  if True, return a trainied classifier to be evaluated
                    
        """
        super().__init__()

        self.layer = layer
        self.fully_supervised = fully_supervised

        self.l0 = nn.Linear(512*20*20, 512)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256,64)
        # single hidden layer classifier
        self.l3 = nn.Linear(64,200)
        self.l4 = nn.Linear(200,10)

        self.b0 = nn.BatchNorm2d(512)
        self.b1 = nn.BatchNorm2d(256)
        self.b2 = nn.BatchNorm2d(64)

        self.d3 = nn.Dropout(0.5)

        self.encoder = Encoder()
                
            
        if eval==False:
            if not fully_supervised:
                # load trained encoder parameter
                if params_file is None:
                    raise ValueError("[!]params_file is not given. Can not load the trained encoder.")
                else:
                    params_file = torch.load(str(params_file))
                    new_params_file = {}
                    for key, value in params_file.items():
                        # To remove the "module" in the key name "module.c0.weight" 
                        keys = key.split('.')[1:]
                        keys = ".".join(keys)
                        new_params_file[keys] = value
                    self.encoder.load_state_dict(new_params_file)


    def forward(self, x):
        if self.fully_supervised:
            _,_, Y = self.encoder(x)
            h = Y.detach()
        else:
            if self.layer=='conv':
                batch_size = x.shape[0]
                M, _, _ = self.encoder(x)
                M = M.detach()
                h = F.relu(self.l0(M.view(batch_size,-1)))
                h = F.relu(self.l1(h))
                h = self.l2(h)
            elif self.layer=='fc':
                _, fc, _ = self.encoder(x)
                fc = fc.detach()
                h = self.l2(fc)
            else: #self.layer=='Y'
                _, _, Y = self.encoder(x)
                Y = Y.detach()
                h = Y
        # A single hidden layer classifier with 200 units and drop out 
        h = F.relu(self.d3(self.l3(h)))
        prediction = F.softmax(self.l4(h), dim=1)

        return prediction

