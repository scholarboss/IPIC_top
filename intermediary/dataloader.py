import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
from utils import *
import lmdb
import h5py
import struct

mode=0     # 0, 1, 2
SNRdb=10   # 8 -- 12


class ChannelDataLoader(Dataset):
    def __init__(self, data, pilot=8):
        super(ChannelDataLoader, self).__init__()
        self.pilot = pilot
        self.data = data
        if pilot == 8:
            self.pilot_num = np.array([1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0])
        elif pilot == 32:
            self.pilot_num = np.array([0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0])

    def __getitem__(self, idx):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X=[bits0, bits1]
        HH = self.data[idx]
        mode = np.random.randint(0, 3)
        SNRdb = np.random.choice(np.arange(8, 12.1, 0.4))
        # mode = 0
        # SNRdb = 10
        YY = MIMO(X, HH, SNRdb, mode, self.pilot)/20 ###
        XX = np.concatenate((self.pilot_num, bits0, bits1), 0)
        return torch.tensor(YY).float(), torch.tensor(XX).float()

    def __len__(self):
        return len(self.data)


class LmdbDataLoader(Dataset):
    def __init__(self, pilot=8, repeat=20):
        super(LmdbDataLoader, self).__init__()
        self.pilot = pilot
        # self.data = data
        self.repeat = repeat
        dir = 'data/Pilot_{}_{}'.format(pilot, repeat)
        env_db = lmdb.Environment(dir)
        self.txn = env_db.begin()
        if pilot == 8:
            self.pilot_num = np.array([1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0])
        elif pilot == 32:
            self.pilot_num = np.array([0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0])

    def __getitem__(self, idx):
        idx_in = 'In_{}'.format(idx)
        idx_out = 'Out_{}'.format(idx)
        buf_in = self.txn.get(idx_in.encode('ascii'))
        buf_out = self.txn.get(idx_out.encode('ascii'))
        value_in = np.frombuffer(buf_in)
        value_out = np.frombuffer(buf_out, dtype=np.bool)
        value_out = np.concatenate((self.pilot_num, value_out), 0)

        return torch.tensor(value_in).float(), torch.tensor(value_out).float()

    def __len__(self):
        return 300000*self.repeat





class NpyDataLoader(Dataset):
    def __init__(self, data, pilot=8):
        super(NpyDataLoader, self).__init__()
        self.pilot = pilot
        self.data = data
        # self.num = num
        if pilot == 8:
            self.pilot_num = np.array([1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0])
        elif pilot == 32:
            self.pilot_num = np.array([0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0])
         

    def __getitem__(self, idx):

        value_in = self.data['x'][idx]
        value_out = self.data['y'][idx]
        value_out = np.concatenate((self.pilot_num, value_out), 0)

        return torch.tensor(value_in).float(), torch.tensor(value_out).float()

    def __len__(self):
        return len(self.data['x'])



class H5pyDataLoader(Dataset):
    def __init__(self, pilot=8, num=0):
        super(H5pyDataLoader, self).__init__()
        self.pilot = pilot
        # self.data = data
        self.num = num
        if pilot == 8:
            self.pilot_num = np.array([1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0])
        elif pilot == 32:
            self.pilot_num = np.array([0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0])
        self.name = 'data/H5py_Pilot_{}/Data_{}.hdf5'.format(pilot, num)
        with h5py.File(self.name, 'r') as f:
            self.len = len(f['x'])
        self.data = None
         

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(self.name, 'r')

        value_in = self.data['x'][idx]
        value_out = self.data['y'][idx]
        value_out = np.concatenate((self.pilot_num, value_out), 0)

        return torch.tensor(value_in).float(), torch.tensor(value_out).float()

    def __len__(self):
        return self.len



class BinDataLoader(Dataset):
    def __init__(self, X, Y, pilot=8):
        super(BinDataLoader, self).__init__()
        self.pilot = pilot
        self.X = X
        self.Y = Y
        self.len = len(Y)//1024
        if pilot == 8:
            self.pilot_num = np.array([1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0])
        elif pilot == 32:
            self.pilot_num = np.array([0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0])

    def __getitem__(self, idx):
        input_beg = idx*2048*4
        input_end = (idx+1)*2048*4
        label_beg = idx*1024
        label_end = (idx+1)*1024
        
        value_in = struct.unpack('f'*2048, self.X[input_beg:input_end])
        value_out = struct.unpack('?'*1024, self.Y[label_beg:label_end])

        value_out = np.concatenate((self.pilot_num, value_out), 0)
        return torch.tensor(value_in).float(), torch.tensor(value_out).float()

    def __len__(self):
        return self.len

