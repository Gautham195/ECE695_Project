#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 3 11:21:30 2024

@author: pmaletti
"""

import argparse
import os
import numpy as np
import math
import sys
import random
from dataloader.dataloader import load_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_gan import Generator, Discriminator
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt 
from dataloader import dataloader_smap
import matplotlib.pyplot as plt 
from utils import EstimateGaussianSigma, LogGaussianProba

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
parser.add_argument("--device", default="cpu", help="device: cuda | cpu - cuda:0 will use only the first gpu")
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

def TestVAE(args, network_name, test_signal, sequence_length, feature_size):
    # test_data = GetDataTest(test_signal, sequence_length, feature_size)
    test_data = test_signal
    #print(test_data.shape)
    network = torch.load(network_name).to(args.device)
    predicted_data = []
    for inp in test_data:
        inp = torch.tensor(inp, dtype=torch.float32)
        inp = torch.unsqueeze(inp, 0)
        inp =inp.to(args.device)
        #print("Input size ",inp.size())
        #inp = inp.to(torch.device("cpu"))
        out = network(inp)
        out = out[0].unsqueeze(-1)
        predicted_data.append(out[0, :, 0].cpu().detach().numpy())
    predicted_data = np.concatenate(predicted_data, axis=0)
    #print(predicted_data.shape)
    return predicted_data



def compute_gradient_penalty(D, real_samples, fake_samples, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.unsqueeze(-1))).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients_ = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients_.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(args, device, generator, discriminator, dataloader,criterion, epochs, optimizer_G, optimizer_D, Tensor, lambda_gp):
    # ----------
    #  Training
    # ----------
    batches_done = 0
    LossD = []
    LossG = []

    generator.to(args.device)
    discriminator.to(args.device)

    for epoch in range(args.n_epochs):
        # print("\n")
        g_losses_per_print_cycle = []           
        d_losses_per_print_cycle = [] 
        for i, data in enumerate(dataloader):


            data_x, data_y = data
            # Configure input
            data_x, data_y = Variable(data_x.type(Tensor)), Variable(data_y.type(Tensor))

            real_labels = torch.ones((data_x.shape[0]), device = device)
            fake_labels = torch.zeros((data_x.shape[0]), device = device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Maximization step
            # Step 1a
            discriminator.zero_grad()

            # Real images
            real_validity = discriminator(data_x).view(-1)
            
            d_loss_real = criterion(real_validity, real_labels)
            d_loss_real.backward()

            # Step 1b

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (data_x.shape[0], args.sequence_length, args.feature_size))))

            # Generate a batch of images
            fake_x = generator(z)[0]
            
            # Fake images
            fake_validity = discriminator(fake_x.detach()).view(-1)
            
            
            d_loss_fake = criterion(fake_validity, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            d_losses_per_print_cycle.append(d_loss)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, data_x.data, fake_x.data, Tensor)
            # Adversarial lossi
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            # d_loss.backward()
            optimizer_D.step()


            #Minimization Step
            #STep 2
            generator.zero_grad()
            output = discriminator(fake_x).view(-1)
            g_loss = criterion(output, real_labels)
            g_losses_per_print_cycle.append(g_loss)
            g_loss.backward()
            optimizer_G.step()

            if batches_done % args.sample_interval == 0:
                print("\n")
                mean_D_loss = torch.mean(torch.FloatTensor(d_losses_per_print_cycle))
                mean_G_loss = torch.mean(torch.FloatTensor(g_losses_per_print_cycle))

                LossD.append(mean_D_loss.item())
                LossG.append(mean_G_loss.item())
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(dataloader), mean_D_loss.item(), mean_G_loss.item())
                )

                d_losses_per_print_cycle = []
                g_losses_per_print_cycle = []

            batches_done += 1

    torch.save(generator, os.path.join(args.out, f"Trained_Generator_{args.channel}"))
    torch.save(discriminator, os.path.join(args.out, f"Trained_Discriminator_{args.channel}"))

    return LossG, LossD

# Calculate the step size for dividing the epochs into partitions
# step_size = args.n_epochs / len(LossD)

def main():
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out, exist_ok=True)

    os.makedirs('LossPlots', exist_ok=True)

    os.makedirs('PredictedSignal', exist_ok=True)

    # Set seed 
    set_seed(args.seed)

    # Set device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Loss weight for gradient penalty
    lambda_gp = 10

    #initialize generator and discriminator
    generator = Generator(args.sequence_length, args.hidden_size, args.feature_size, args.latent_dim)
    discriminator = Discriminator(args.sequence_length, args.hidden_size, args.feature_size)

    print("\n", generator)
    print("\n", discriminator)

    # Configure data loader
    train_dataloader = load_data(args, 'Train')

    # optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    criterion = nn.BCEWithLogitsLoss()

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # Train Generator and Discriminator
    LossG, LossD = train(args, device, generator, discriminator, train_dataloader, criterion, args.n_epochs, optimizer_G, optimizer_D, Tensor, lambda_gp)

    # Plot loss
    epochs = range(1, args.n_epochs+1)

    plt.figure(figsize=(5,5))    
    plt.title(f"Generator and Discriminator Loss During Training - {args.channel}")    
    plt.plot(LossD, label='Discriminator')
    plt.plot(LossG, label='Generator')
    plt.xlabel("iterations")   
    plt.ylabel("Loss")         
    plt.legend()
    plt.savefig(f'./LossPlots/DvG_Loss_test_LinearLayers_{args.channel}.png')

    test_signal, test_signal_ = load_data(args, 'Test') 
    predicted_signal = TestVAE(args, f'./ckpts/Trained_Generator_{args.channel}', test_signal, args.sequence_length, args.feature_size)
    predicted_signal.reshape(test_signal.shape)


    # #### Plot signals  ####
    plt.figure()
    start = 6500; end = 7500
    # start = 4500; end = 5500
    # start = 2000; end = 3000
    plt.plot(predicted_signal[start:end])  # Plot only the part of test_signal corresponding to predicted_data
    plt.plot(test_signal_[start:end])
    plt.legend()
    # plt.show()
    plt.savefig(f'./PredictedSignal/Predicted_{args.channel}.png', dpi=200) 


    ''' Commented out because generated signal not good enough to gte error plot '''
    # e = np.array([abs(predicted_signal[i] - test_signal[i]) for i in range(0, len(test_signal))])
    # mean, var = EstimateGaussianSigma(e, 0, len(test_signal))
    # log_gaussian_e = LogGaussianProba(mean, var)

    # plt.figure()
    # plt.plot(log_gaussian_e, label = "Error liklihood")
    # plt.legend()
    # plt.savefig(f'Anomaly_{args.channel}.png', dpi=200) 

    # e = np.array([abs(predicted_signal[i] - test_signal[i]) for i in range(0, len(test_signal))])
    # mean, var = EstimateGaussianSigma(e, 0, len(test_signal))
    # log_gaussian_e = LogGaussianProba(mean, var)

if __name__=='__main__':
    main()

