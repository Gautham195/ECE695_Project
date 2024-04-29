# Anomaly Detection with Time Series Data on the SMAP dataset

This repository was created by Gautham Reddy Prakash (gprakas@purdue.edu) and Prajna Malettira (pmaletti@purdue.edu) for the Final project of ECE 69500 Generative Models at Purdue University. 



## Introduction
Anomaly detection is the detection of rare occurrences that are different from the established pattern of behaviours. In the real world applications, these anomalies signify erroneous system behaviour which must be detected quickly to avoid system failure. Hence is it vital that they are detected easily.


![introplot](https://github.com/pmaletti/GM_Project/blob/main/images/Idea.png?raw=true)



## Dataset
The SMAP dataset is a dataset contains expert-labelled time series anomaly data. It is collected by NASA from the Soil Moisture Active Passive (SMAP) satellite containing numerous channels, where each channel represents different time series data. 
Each channel contains a train signal which has no anomalies and a test signal with anomalies



## Idea

![ideaplot](https://github.com/pmaletti/GM_Project/blob/main/images/IntroPics.png?raw=true)



## Implementation
1) The anomaly signal is passed through a VAE to produce a signal that is then subtracted from the anomaly signal. Each sample is then clustered into two clusters(one for anomalies, and one for normal data). Note that while training the VAE, the training signal has NO anomalies in it. 

The VAE draws from a distribution and hence, it is not severely affected by anomalies. This is the intuition behind using a VAE. 

![Implementation](https://github.com/pmaletti/GM_Project/blob/main/images/Implementation.png?raw=true)

2) We also tried to do the same but with a Generative Adverserial Network (GAN). The training procedure of the GAN is shown below - 
![Implementation2](https://github.com/pmaletti/GM_Project/blob/main/images/TrainingProcess.png?raw=true)

The GAN training procedure and architecture was inspired by the [TADGAN](https://ieeexplore.ieee.org/abstract/document/9378139) paper. It performs the same function as a VAE, but the only difference is that it was trained in a Generative Adverserial way.


## Novelty 
1) Many papers use GANs and VAEs for anonaly detection. But we took ideas from two papers to implement our project - 
[Hundman et all](https://arxiv.org/pdf/1802.04431) and [Anomaly Detection with Conditional Variational Autoencoders"](https://ieeexplore.ieee.org/abstract/document/8999265)

2) In Hundman's paper, they used a predictor approach, where they predict the next samples from the previous samples useing LSTMs. This predictor is not a generative model. We replaced this LSTM predictor with a generative model, specifically, a VAE/GAN to reconstruct the signal without anomalies. We got the idea of using a VAE from the "Anomaly Detection with Conditional Variational Autoencoders" paper. 

3) Hundman uses thresholding to find his anomalies. To make our approach novel, we also clustered (using K-means clustering) the samples into anomalous and normal data. 


## Results for VAE implementation
1) Variational Auto Encoder results

The output of the VAE is compared to the actual signal in the leftmost figure. The right-most figure has the anomaly that was detected from this particular sequence. 

![results](https://github.com/pmaletti/GM_Project/blob/main/images/Results.png?raw=true)

2) Generative Adverserial Networks (GAN) results -

The figures below show the best results we obtained using the TADGAN network. However, the results were poor and we did not proceed with the detection of anomalies since the reconstruction was bad. 

![results2](https://github.com/pmaletti/GM_Project/blob/main/images/ResultsGAN.png?raw=true)

## Evaluation
```
cd ProjectECE695
./scripts/VAE.sh
./scripts/GAN.sh
```
To run the VAE, use ./scripts/VAE.sh and to run the GAN, use ./scripts/GAN.sh
The output plots must be in the folder ProjectECE695


Thank you! :)