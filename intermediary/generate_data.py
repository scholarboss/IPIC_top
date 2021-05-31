import numpy as np
import lmdb
import os
import sys
import pickle
from tqdm import tqdm
import struct
from utils import *
import argparse
import h5py


data1=open('data/H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]

Htest=H[300000:,:,:]
H=H[:300000,:,:]


def save_h5(h5f, data, target):
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = None  # 设置数组的第一个维度是0
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old = dataset.shape[0]
    len_new = len_old + data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))  # 修改数组的第一个维度
    dataset[len_old:len_new] = data  # 存入新的文件


def MIMO2LMDB(pilot=8, repeat=120):


    Lmbd_name = 'data/Pilot_{}_{}'.format(pilot, repeat)
    # if os.path.exists(Lmbd_name):
    #     print('Folder [{:s}] already exists. Exit...'.format(Lmbd_name))
    #     sys.exit(1)
    length = 300000 * 100
    env = lmdb.open(Lmbd_name, map_size=length*2048*5) #map_size=214748364800)
    txn = env.begin(write=True)
    
    print('processing data... ')
    for i in range(repeat):
        print('dealing [{}]/[{}]...'.format(i+1, repeat))
        for j in tqdm(range(len(H))):
            key_in = 'In_{}'.format(i*len(H)+j)
            key_out = 'Out_{}'.format(i*len(H)+j)
            key_in = key_in.encode('ascii')
            key_out = key_out.encode('ascii')

            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            HH = H[j]
            mode = np.random.randint(0, 3)
            SNRdb = np.random.choice(np.arange(8, 12.1, 0.4))
            YY = MIMO(X, HH, SNRdb, mode, pilot)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            XX = np.array(XX, dtype=np.bool)

            txn.put(key_out, XX)
            txn.put(key_in, YY)
            
        txn.commit()
        txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('\nFinish writing lmdb...')

def getnpz(pilot=8, repeat_begin=0, repeat_end=15):
    npy_dir = 'data/NumPy_Pilot_{}'.format(pilot)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    
    print('processing data... ')

    for i in range(repeat_begin, repeat_end):
        print('dealing [{}]/[{}]...'.format(i+1, repeat_end-repeat_begin))

        input_labels = []
        input_samples = []
        for j in tqdm(range(len(H)//100)):
        # for j in tqdm(range(200)):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            mode = np.random.randint(0, 3)
            SNRdb = np.random.choice(np.arange(8, 12.1, 0.4))
            YY = MIMO(X, HH, SNRdb, mode, pilot)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            XX = np.array(XX, dtype=np.bool)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        save_name = os.path.join(npy_dir, 'Data_{}.npz'.format(i))
        np.savez(save_name, x=batch_y, y=batch_x)
        del batch_x
        del batch_y
        del input_samples
        del input_labels

    print('\nFinish writing npz...')

def getH5py(pilot=8, repeat_begin=0, repeat_end=15):
    npy_dir = 'data/H5py_Pilot_{}'.format(pilot)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    
    print('processing data... ')
    
    for i in range(repeat_begin, repeat_end):
        save_name = os.path.join(npy_dir, 'Data_{}.hdf5'.format(i))
        f = h5py.File(save_name, 'w')
        print('dealing [{}]/[{}]...'.format(i+1, repeat_end-repeat_begin))


        for _ in range(1):
            input_labels = []
            input_samples = []
            for j in tqdm(range(len(H))):
            # for j in tqdm(range(200)):
                bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
                bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
                X=[bits0, bits1]
                temp = np.random.randint(0, len(H))
                HH = H[temp]
                mode = np.random.randint(0, 3)
                SNRdb = np.random.choice(np.arange(8, 12.1, 0.4))
                YY = MIMO(X, HH, SNRdb, mode, pilot)/20 ###
                XX = np.concatenate((bits0, bits1), 0)
                XX = np.array(XX, dtype=np.bool)
                input_labels.append(XX)
                input_samples.append(YY)
            batch_y = np.asarray(input_samples)
            batch_x = np.asarray(input_labels)
            save_h5(f, batch_x, 'y')
            save_h5(f, batch_y, 'x')
            # np.savez(save_name, x=batch_y, y=batch_x)
            del batch_x
            del batch_y
            del input_samples
            del input_labels
        f.close()
    print('\nFinish writing npz...')


def getBIN(pilot=8, repeat_begin=0, repeat_end=15):
    Bin_dir = 'data/BIN_Pilot_{}'.format(pilot)
    if not os.path.exists(Bin_dir):
        os.makedirs(Bin_dir)
    
    print('processing data... ')
    
    for i in range(repeat_begin, repeat_end):
        label_name = os.path.join(Bin_dir, 'label_{}.bin'.format(i))
        input_name = os.path.join(Bin_dir, 'input_{}.bin'.format(i))
        # f = h5py.File(save_name, 'w')
        print('dealing [{}]/[{}]...'.format(i+1, repeat_end-repeat_begin))


        for _ in range(1):
            input_labels = []
            input_samples = []
            for j in tqdm(range(len(H))):
            # for j in tqdm(range(200)):
                bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
                bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
                X=[bits0, bits1]
                temp = np.random.randint(0, len(H))
                HH = H[temp]
                mode = np.random.randint(0, 3)
                SNRdb = np.random.choice(np.arange(8, 12.1, 0.4))
                YY = MIMO(X, HH, SNRdb, mode, pilot)/20 ###
                XX = np.concatenate((bits0, bits1), 0)
                XX = np.array(XX, dtype=np.bool)
                input_labels.append(XX)
                input_samples.append(YY)
            batch_y = np.asarray(input_samples, dtype=np.float32)
            batch_x = np.asarray(input_labels, dtype=np.bool)
            batch_x.tofile(label_name)
            batch_y.tofile(input_name)
            # save_h5(f, batch_x, 'y')
            # save_h5(f, batch_y, 'x')
            # np.savez(save_name, x=batch_y, y=batch_x)
            del batch_x
            del batch_y
            del input_samples
            del input_labels
        # f.close()
    print('\n Finish writing BIN...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', type=int, default=8, help='8 or 32')
    parser.add_argument('--repeat_begin', type=int, default=0, help='...')
    parser.add_argument('--repeat_end', type=int, default=15, help='...')
    args = parser.parse_args()

    getBIN(pilot=args.pilot, repeat_begin=args.repeat_begin, repeat_end=args.repeat_end)

    '''
        r = np.load("result.npz")
        r["x"] # 数组a
    '''