import argparse
import os
import numpy as np
import math
import sys
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataloader.dataloader import load_data
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt 
from dataloader import dataloader_smap
import matplotlib.pyplot as plt 


def EstimateGaussianSigma(e_s, start, win_s):
    mean_es = np.mean(e_s[start:start+win_s])
    var_es =  np.var(e_s[start:start+win_s])
    return mean_es,var_es

def LogGaussianProba(mean, variance, x_vector):
    return -(x_vector-mean)*(x_vector-mean) / ((2*variance)+0.00000000001)

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_divergence /= x.size(0)  # Average over batch   
    total_loss = recon_loss + kl_divergence
    return total_loss