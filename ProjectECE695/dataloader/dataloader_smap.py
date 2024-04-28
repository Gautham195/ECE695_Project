import numpy as np
from torch.utils.data import Dataset
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

def NormalizeData(dataMatrix):
    scaler= MinMaxScaler(feature_range=(0,1))
    numCh = dataMatrix.shape[0]
    for i in range(numCh):
        dataMatrix[i,:]=np.reshape(scaler.fit_transform(dataMatrix[i,:][:,None]),(dataMatrix.shape[1]))
    return dataMatrix

def GetData(signal, sequence_length ,feature_size, bench_type):
    if bench_type=='Train':
        dataX, dataY=[],[]
        #Note that since this is a VAE, the input = output
        for i in range(signal.shape[0] - sequence_length):
            dataX.append(signal [i : (i + sequence_length),:])
            dataY.append(signal [i : (i + sequence_length),:])
            
        return np.array(dataX),np.array(dataY)
    else:
        dataX = [signal[i : i + sequence_length, :] for i in range(0, signal.shape[0] - sequence_length, sequence_length)]
        return np.array(dataX)

def GetAndPreprocessData(file_name, bench_type):
    if bench_type=='Train':
        TRMdataMatrix = np.load('./dataset/SMAP_sml/train/'+file_name+'.npy')
    else:
        TRMdataMatrix = np.load('./dataset/SMAP_sml/test/'+file_name+'.npy')
    RMdataMatrix = np.transpose(TRMdataMatrix)
    MdataMatrix = RMdataMatrix
    NormalizedMatrix = NormalizeData( MdataMatrix ) # Normalize the data to [0,1]
    train_data = NormalizedMatrix.transpose()[:,0]     #Split the 1 dimension of data
    #use med value filter to denoising first 
    train_data = signal.medfilt(train_data.reshape(train_data.shape[0]),5)
    train_data = train_data.reshape(train_data.shape[0],1)
    
    return train_data

class DataSetToLoadTimeSeries(Dataset):
    
    def __init__(self, dataX, dataY=None, bench_type=None):
        super().__init__()
        self.dataX = dataX
        self.bench_type = bench_type
        if bench_type=='Train':
            self.dataY = dataY
        
    def __len__(self):
        #Must return the number of training samples
        return self.dataX.shape[0]
    
    def __getitem__(self, index):
        #Returns a tensor of X and Y
        X = self.dataX[index, :, 0]
        X = torch.tensor(X)
        X = torch.unsqueeze(X, 1)
        if self.bench_type=='Train':
            Y = self.dataY[index, :, 0]
            
            Y = torch.tensor(Y)
            Y = torch.unsqueeze(Y, 1)
            return X,Y
        else:
            return X


def GetTestData(file_name):
    TRMdataMatrix = np.load('./dataset/SMAP_sml/test/'+file_name+'.npy')
    RMdataMatrix = np.transpose(TRMdataMatrix)
    MdataMatrix = RMdataMatrix
    NormalizedMatrix = NormalizeData( MdataMatrix ) # Normalize the data to [0,1]
    test_data = NormalizedMatrix.transpose()[:,0]     #Split the 1 dimension of data
    #use med value filter to denoising first 
    test_data = signal.medfilt(test_data.reshape(test_data.shape[0]),5)
    test_data = test_data.reshape(test_data.shape[0],1)
    
    return test_data

