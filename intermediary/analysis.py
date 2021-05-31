import numpy as np
import matplotlib.pyplot as plt
import struct
from utils import *

mode=0         # 0 1 2 
SNRdb=100       # 8 -- 12
Pilotnum=8    # 32 or 8
###########################以下仅为信道数据载入和链路使用范例############

data1=open('H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*2000,data1.read(4*2*2*2*32*2000))
H1=np.reshape(H1,[2000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]

# Htest=H[300000:,:,:]
# H=H[:300000,:,:]


####################使用链路和信道数据产生训练数据##########
def generator(batch,H):
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            YY = MIMO(X, HH, SNRdb, mode,Pilotnum)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        
        yield (batch_y, batch_x)

########产生测评数据，仅供参考格式##########
def generatorXY(batch, H):
    input_labels = []
    input_samples = []
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    return batch_y, batch_x
    



Y1 = open('Y_1.csv', 'r')
Y2 = open('Y_2.csv', 'r')

for i in range(1000):
    line1 = Y1.readline().split(',')
    line1 = np.array(line1, dtype=np.float)
    hist1, edges1 = np.histogram(line1, bins=50)
    hist1, edges1 = np.array(hist1), np.array(edges1)[:-1]

    line2 = Y2.readline().split(',')
    line2 = np.array(line2, dtype=np.float)
    hist2, edges2 = np.histogram(line2, bins=50)
    hist2, edges2 = np.array(hist2), np.array(edges2)[:-1]
    # plt.plot(edges2, hist2)
    # plt.show()

    # 仿真数据
    Y, X = generatorXY(1, H)
    Y, X = Y[0], X[0]
    hist3, edges3 = np.histogram(Y, bins=50)
    hist3, edges3 = np.array(hist3), np.array(edges3)[:-1]

    # 可视化
    fig, ax = plt.subplots(1,2, figsize=(18, 9))
    ax[0].plot(edges1, hist1, label="Y_1")
    ax[0].plot(edges2, hist2, label="Y_2")
    ax[0].plot(edges3, hist3, label="Y_3")
    ax[0].legend()

    axis = list(range(len(line1)))
    ax[1].scatter(axis, line1, label="Y_1", s=[5 for _ in range(len(axis))])
    ax[1].scatter(axis, line2, label="Y_2", s=[5 for _ in range(len(axis))])
    ax[1].scatter(axis, Y, label="Y_3",  s=[5 for _ in range(len(axis))])
    ax[1].legend()    

    plt.show()

