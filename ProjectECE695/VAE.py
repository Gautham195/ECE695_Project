#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:11:41 2024

@author: gauthamreddyprakash
"""
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
from torch.utils.data import DataLoader
import scipy.signal as signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_vae import VAE
import utils
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# from dataloader import dataloader_smap
from dataloader.dataloader import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="set random seed")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=50, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=10, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--dataroot", default="", help="path to dataset")
parser.add_argument("--dataset", default="SMAP", help="folder | cifar10 | mnist | stl")
parser.add_argument("--device", default="cuda", help="device: cuda | cpu - cuda:0 will use only the first gpu")
parser.add_argument("--name", default="test")
parser.add_argument("--out", default="ckpts", help="checkpoint directory")
parser.add_argument("--channel", default="A-3", help="channel being used")
parser.add_argument("--sequence_length", default="200", type=int, help="sequence length")
parser.add_argument("--hidden_size", default="150", type=int, help="hidden size")
parser.add_argument("--feature_size", default="1", type=int, help="feature size")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def CreateVAE(args):
    vae_net = VAE(args.sequence_length, args.hidden_size, args.feature_size, args.latent_dim)
    decay_rate = args.lr/100;
    optimizer = torch.optim.Adam(vae_net.parameters(), lr=args.lr, betas = (0.9 , 0.999), eps = 1e-08, 
                                 weight_decay = decay_rate);
    #Decalring the loss function - MSELoss
    loss_function = vae_loss;#torch.nn.MSELoss()
    
    return (vae_net, optimizer, loss_function)

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_divergence /= x.size(0)  # Average over batch   
    total_loss = recon_loss + kl_divergence
    return total_loss

def remove_ones_below_threshold(sequence, min_threshold):
    num_zeros = np.count_nonzero(sequence == 0)
    num_ones = np.count_nonzero(sequence == 1)
    
    # Compare the counts
    if num_ones > num_zeros:
        not_anomaly = 1
        anomaly = 0
    else:
        anomaly = 1
        not_anomaly = 0

    current_count = 0
    for i in range(len(sequence)):
        if sequence[i] == anomaly:
            current_count += 1
        else:
            if current_count < min_threshold:
                sequence[i - current_count:i] = [not_anomaly] * current_count
            current_count = 0
    return sequence
    
def TrainVAE(args, network, device, optimizer, loss_function, dataloader):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device)
    # network.to(device)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    criterion = loss_function
    # trainX, trainY = GetData(train_signal, sequence_length, feature_size)
    # mydataset_training = DataSetToLoadTimeSeries(trainX, trainY)
    # train_data_loader = DataLoader(mydataset_training, batch_size = batch_size, shuffle = shuffle)
    
    for epoch in range ( args.n_epochs ):
        #Starting the training part - 
        network.train()
        total_loss = 0
        for i , data in enumerate ( dataloader ):
            inputs , y = data
            #print(inputs.dtype);
            inputs = inputs . to (torch.float32). to ( device )
            y = y . to (torch.float32). to ( device )
            #print(inputs.dtype);
            optimizer . zero_grad ()
            outputs = network ( inputs )
            #Get outputs
            reconstructed_x = outputs[0]
            mu = outputs[1]
            sigma = outputs[2]
            reconstructed_x.to(device)
            y.to(device)
            mu.to(device)
            sigma.to(device)
            loss = criterion ( reconstructed_x, y, mu, sigma)
            #print(y.shape)
            #print(reconstructed_x.shape)
            #print(loss.data);
            del inputs
            del y
            loss . backward ()
            #Just for printing the loss
            total_loss += loss.item()
            optimizer.step()
        lr_scheduler.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, total_loss / len(dataloader)))
        
    return network


def TestVAE(args, network_name, test_signal):
    # test_data = GetDataTest(test_signal, args.sequence_length, args.feature_size)
    #print(test_data.shape)
    test_data = test_signal
    network = torch.load(network_name)
    predicted_data = []
    for inp in test_data:
        inp = torch.tensor(inp, dtype=torch.float32)
        inp = torch.unsqueeze(inp, 0)
        #print("Input size ",inp.size())
        #inp = inp.to(torch.device("cpu"))
        out = network(inp)
        predicted_data.append(out[0][0, :, 0].detach().numpy())
    predicted_data = np.concatenate(predicted_data, axis=0)
    #print(predicted_data.shape)
    return predicted_data

def Plot(predicted_signal, test_signal, e, cluster_labels, start, end):
    plt.figure()
    start = 6000; end = 8000
    #Plot()
    plt.plot(predicted_signal[start:end], label = "Prediction")  # Plot only the part of test_signal corresponding to predicted_data
    plt.plot(test_signal[start:end], label = "Original")
    plt.xlabel("Sample")
    plt.ylabel("Signal value")
    plt.title("Reconstructed signal vs original signal")
    plt.legend()
    plt.savefig('test.png', dpi=200) 
    plt.figure()
    plt.plot(e[start:end], label = "Error signal")
    plt.title("Error signal")
    plt.xlabel("Sample")
    plt.ylabel("Signal value")
    plt.legend()
    plt.savefig("Error")
    plt.figure()
    plt.plot(cluster_labels, label = "Sequence")
    plt.xlabel("sample")
    plt.ylabel("Anomaly presence")
    plt.title("Anomaly sequence")
    #plt.figure()
    #plt.plot(log_gaussian_e)
    plt.savefig("Seq.png", dpi=200)
        


def train(args, is_train, network, dataloader, optimizer, loss_function, device, model_path=None):
        
    network = network.to(device)
    if is_train:
        network = TrainVAE(args, network, device, optimizer, loss_function, dataloader)
    
    else:    
        predicted_signal = TestVAE(args, model_path, dataloader)

    if is_train:
        return network
    else:
        return predicted_signal

def get_clusters(log_gaussian_e):

    log_gaussian_e = np.array(log_gaussian_e)

    log_gaussian_e = log_gaussian_e.reshape(-1, 1)

    num_clusters = 2

    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(log_gaussian_e)

    cluster_labels = kmeans.labels_
    cluster_labels = remove_ones_below_threshold(cluster_labels, 30)

    return cluster_labels

def main():

    args = parser.parse_args()
    print(args)

    # Set seed 
    set_seed(args.seed)

    # train_signal = GetAndPreprocessData(channels[0])
    network, optimizer, loss_function = CreateVAE(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # trainX, trainY = GetData(train_signal, sequence_length, feature_size)
    # mydataset_training = DataSetToLoadTimeSeries(trainX, trainY)
    # train_data_loader = DataLoader(mydataset_training, batch_size = batch_size, shuffle = shuffle)

    #Train network
    train_data_loader = load_data(args, 'Train')
    network = train(args, True, network, train_data_loader, optimizer, loss_function, device)
    network = network.to(torch.device("cpu"))
    torch.save(network, f"{args.out}/Trained_VAE")

    #Test network
    # test_signal = GetTestData(channels[0])
    test_signal, test_signal_ = load_data(args, 'Test')
    predicted_signal = train(args, False, network, test_signal, optimizer, loss_function, device, f"{args.out}/Trained_VAE")

    #Get error signal
    e = np.array([abs(predicted_signal[i] - test_signal_[i]) for i in range(0, len(predicted_signal))])
    mean, var = utils.EstimateGaussianSigma(e, 0, len(test_signal))
    log_gaussian_e = utils.LogGaussianProba(mean, var, e)

    cluster_labels = get_clusters(log_gaussian_e)

    #Plot results
    start = 4000; end = 5000
    Plot(predicted_signal, test_signal_, e, cluster_labels, start, end)

if __name__=='__main__':
    main()

