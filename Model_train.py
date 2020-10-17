#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
If you have any questions, please contact me with https://github.com/xufana7/AutoEncoder-with-pytorch
Author, Fan xu Aug 2020
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE, NMSE_cuda, print_network, NMSE_torch
import os
import torch.nn as nn
import sys
import argparse
# python Model_train.py 64 1000


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1.0e-03, help='learning rate')
parser.add_argument('--encoder', type=str, default='none', help='encoder checkpoint')
parser.add_argument('--decoder', type=str, default='none', help='decoder checkpoint')
parser.add_argument('--decay', type=int, default=3000, help='learning rate decay')
parser.add_argument('--start', type=int, default=0, help='start epoch')
parser.add_argument('--end', type=int, default=3000, help='end epoch')
parser.add_argument('--channel', type=int, default=64, help='model internal channel')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--tag', type=str, default='', help='tag on checkpoint')
parser.add_argument('--loss', type=str, default='mse', help='mse or nmse')

args = parser.parse_args()




# Parameters for training
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = args.batchsize
# epochs = 1000
learning_rate = args.lr
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

# Model construction
model = AutoEncoder(feedback_bits, channel=args.channel)
print_network(model)

# load model
if args.encoder != 'none':
    encoder_path = './Modelsave/{}.pth.tar'.format(args.encoder)
    model.encoder.load_state_dict(torch.load(encoder_path)['state_dict'])
if args.decoder != 'none':    
    decoder_path = './Modelsave/{}.pth.tar'.format(args.decoder)
    model.decoder.load_state_dict(torch.load(decoder_path)['state_dict'])


if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()
L1loss = nn.L1Loss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address + '/H_train.mat', 'r')
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
# data = np.transpose(mat['H_train'][:,:50])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(70%) and validation(30%)
np.random.seed(0)
np.random.shuffle(data)
start = int(data.shape[0] * 0.9)
x_train, x_test = data[:start], data[start:]

# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for testing
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
best_loss = 999999999999
for epoch in range(args.start, args.end):
    # model training
    model.train()
    if epoch % args.decay == 0:
        print('learning rate decay from {:.8f} to {:.8f}'.format(learning_rate, learning_rate*0.5))
        learning_rate = learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    # measure data loading time    
    for i, input_ in enumerate(train_loader):
        # adjust learning rate

        input_ = input_.cuda()

        # compute output
        output = model(input_)
        # loss = NMSE_cuda(input_, output, mode='sum')
        if args.loss == 'mse':
            loss = criterion(output, input_)
        elif args.loss == 'nmse':
            loss = NMSE_torch(input_, output)
        elif args.loss == 'L1loss':
            loss = L1loss(output, input_)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % print_freq == 0:
            print('Train:> Epoch: [{0}][{1}/{2}]\t'
                  'lr {lr:.8f} | Loss {loss:.6f}\t'.format(
                epoch, i+1, len(train_loader), lr=learning_rate, loss=loss.item()))
    model.eval()
    # total_loss = 0
    with torch.no_grad():
        for i, input_ in enumerate(test_loader):
            # convert numpy to Tensor
            input_ = input_.cuda()
            output = model(input_)
            # total_loss += NMSE_cuda(input_, output, mode='mean').item() * input_.size(0)
            output = output.cpu().numpy()
            if i == 0:
                y_test = output
            else:
                y_test = np.concatenate((y_test, output), axis=0)
        nmse = NMSE(np.transpose(x_test, (0, 2, 3, 1)), np.transpose(y_test, (0, 2, 3, 1)))
        # average_loss = total_loss/10 #len(test_loader)
        print('Test:> Epoch: {} nmse {:.6f}'.format(
                epoch, nmse))
        if nmse < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder_{}{}.pth.tar'.format(args.channel, args.tag)
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder_{}{}.pth.tar'.format(args.channel, args.tag)
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print("Model saved")
            # best_loss = average_loss
            best_loss = nmse




