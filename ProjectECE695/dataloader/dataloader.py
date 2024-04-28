"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from dataloader import dataloader_smap 

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt,bench_type='Train'):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET

    if opt.dataset in ['SMAP']:
        signal = dataloader_smap.GetAndPreprocessData(opt.channel, bench_type)
        if bench_type=='Train':
            train_X, train_Y = dataloader_smap.GetData(signal, opt.sequence_length, opt.feature_size,bench_type)
            ds = dataloader_smap.DataSetToLoadTimeSeries(train_X, train_Y, bench_type='Train')
        else:
            test_X = dataloader_smap.GetData(signal, opt.sequence_length, opt.feature_size,bench_type)

    else:
        raise NotImplementedError

    ## DATALOADER
    if opt.dataset in ['SMAP']:
        if bench_type=='Train':
            dl = DataLoader(dataset=ds, batch_size=opt.batch_size, shuffle=True, drop_last=True)

            return dl
        else:
            return test_X, signal
    else:
        train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)

        return Data(train_dl, valid_dl)
