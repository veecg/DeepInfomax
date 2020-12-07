# Three loss function for DIM
# - DV
# - JSD
# - infoNCE
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ALPHA, BETA, GAMMA
from models import Encoder, GlobalDIM, LocalDIM, PriorDiscriminator

def MI_JSD(Tp, Tn):
    """
    Input:
        Tp: Discriminator score of positive pair
        Tn: Discriminator score of negative pair
    """
    Ej = -F.softplus(-Tp).mean()
    Em = F.softplus(Tn).mean()
    I = Ej - Em
    return I


class JSD_Loss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if alpha!=0:
            self.global_dim = GlobalDIM()
        if beta!=0:
            self.local_dim = LocalDIM()
        if gamma!=0:
            self.prior_mat = PriorDiscriminator()
        
    def forward(self, Y, M):
        # create positive and negative pairs; One image has 1 negative sample here
        M_prime = torch.cat((M[1:],M[0].unsqueeze(0)),dim=0)
        
        if self.alpha!=0:
            Tp = self.global_dim(Y, M)
            Tn = self.global_dim(Y, M_prime)
            Ig = MI_JSD(Tp,Tn)
            GLOBAL = self.alpha * Ig
        else:
            GLOBAL = 0


        if self.beta!=0:
            Tp = self.local_dim(Y,M)
            Tn = self.local_dim(Y, M_prime)
            Il = MI_JSD(Tp,Tn)
            LOCAL = self.beta * Il
        else:
            LOCAL = 0


        if self.gamma!=0:
            #  Specify prior; in the paper, the authors claim that uniform distribution yields the best result
            prior = torch.rand_like(Y)
            Dp = self.prior_mat(prior)
            Dy = self.prior_mat(Y)
            PRIOR = self.gamma * (torch.log(Dp).mean()+torch.log(1.0-Dy).mean())
        else:
            PRIOR = 0


        if GLOBAL>0:
          print("[!] Global value:", GLOBAL)
        if LOCAL>0:
          print("[!] Local value:", LOCAL)
        if PRIOR>0:
          print("[!] Prior value:", PRIOR)
        return -(GLOBAL+LOCAL+PRIOR)

