#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from gdn import GDN
import common

# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) #/ ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

#'''
# create your own Encoder
class Encoder(nn.Module):
    def __init__(self, feedback_bits=128, M=64, in_chan=2, B = 4):
        super(Encoder, self).__init__()
        out_chan = feedback_bits // B #
        self.head = nn.Sequential(
            nn.Conv2d(in_chan, M, 3, 1, 1),
            common.ResBlock(common.default_conv, M, 3)
        )
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            GDN(M),
            common.ResBlock(common.default_conv, M, 3),
            common.NLResAttModuleDownUpPlus(common.default_conv, M, 3),
            # nn.LeakyReLU(negative_slope=0.3, inplace=True),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=3, stride=2, padding=1, bias=False),
            GDN(M),
            common.TrunkBranch(common.default_conv, M, 3),
            # nn.LeakyReLU(negative_slope=0.3, inplace=True),

            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=3, stride=2, padding=1, bias=False),
            GDN(M),
            common.ResBlock(common.default_conv, M, 3),
            common.NLResAttModuleDownUpPlus(common.default_conv, M, 3),

            common.TrunkBranch(common.default_conv, M, 3)
            # nn.LeakyReLU(negative_slope=0.3, inplace=True),

            # nn.Conv2d(in_channels=M, out_channels=M, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            # GDN(M),
            # nn.Sigmoid()
        )
        self.tail = nn.Conv2d(M, 64, 3, 1, 1)
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(1024, out_chan)
        self.sig = nn.Sigmoid()
        self.quantization = QuantizationLayer(B)

    def forward(self, x):
        out = self.head(x)
        out = self.enc(out)
        out = self.tail(out)
        out = out.view(-1, 1024)
        out = self.sig(self.fc(out))
        return self.quantization(out)

# create your own Decoder
class Decoder(nn.Module):
    def __init__(self, feedback_bits=128, M=64, out_chan=2, B = 4):
        in_chan = feedback_bits // 4
        super(Decoder, self).__init__()
        self.dequantization = DequantizationLayer(B)
        self.fc = nn.Linear(in_chan, 1024)
        # self.dec = nn.Sequential(
        #     nn.Conv2d(in_channels=M*2, out_channels=M*2, kernel_size=3, stride=1, padding=1, bias=False),
        #     GDN(M*4, inverse=True),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),

        #     nn.Conv2d(in_channels=M*4, out_channels=M*2, kernel_size=3, stride=1, padding=1, bias=False),
        #     GDN(M*2, inverse=True),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),

        #     nn.Conv2d(in_channels=M*2, out_channels=M, kernel_size=3, stride=1, padding=1, bias=False),
        #     GDN(M, inverse=True),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),

        #     nn.Conv2d(in_channels=M, out_channels=out_chan, kernel_size=3, stride=1, padding=1, bias=False),
        #     GDN(out_chan, inverse=True),
        #     nn.Sigmoid()
        #     # GDN(M*4),
        #     # nn.LeakyReLU(negative_slope=0.3, inplace=True),
        # )
        self.head = nn.Conv2d(64, M, 3, 1, 1)
        self.dec = nn.Sequential(
            common.TrunkBranch(common.default_conv, M, 3),

            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=3, stride=(1, 2), padding=(1,1), output_padding=(0,1), bias=False),
            common.ResBlock(common.default_conv, M, 3),
            common.NLResAttModuleDownUpPlus(common.default_conv, M, 3),
            
            # nn.LeakyReLU(negative_slope=0.3, inplace=True),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            common.ResBlock(common.default_conv, M, 3),
            common.NLResAttModuleDownUpPlus(common.default_conv, M, 3),

            # nn.LeakyReLU(negative_slope=0.3, inplace=True),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            common.TrunkBranch(common.default_conv, M, 3),
            # GDN(M, inverse=True),
            # nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0,1), bias=False),

            nn.Conv2d(in_channels=M, out_channels=out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dequantization(x)
        out = self.fc(out)
        BatchSize, _ = out.shape
        # print(BatchSize, channel)
        out = out.view(BatchSize, -1, 4, 4)
        out = self.head(out)
        return self.dec(out)
#'''



# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 128 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits, channel=64):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits, M=channel)
        self.decoder = Decoder(feedback_bits, M=channel)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score

def NMSE_torch(x, x_hat):
    x_real = x[:,0,:,:].view(len(x), -1) - 0.5
    x_imag = x[:,1,:,:].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:,0,:,:].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:,1,:,:].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, 1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, 1)
    return torch.mean(mse/power)



# https://github.com/seefun/NAIC_baseline_pytorch
def NMSE_cuda(x, x_hat, mode=None):
    x_real = x[:, 0, :, :].view(len(x),-1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, 1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, 1)
    nmse = mse/power
    if mode == 'sum':
        return torch.sum(nmse)
    if mode == 'mean':
        return torch.mean(nmse)
    return nmse
    
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)



# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
