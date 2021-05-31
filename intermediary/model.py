import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import common
import torch.utils.checkpoint as cp
from collections import OrderedDict



class CNNModel(nn.Module):
    def __init__(self, in_channel=2, out_channel=1, features=64):
        super(CNNModel, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, features, 3, 1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True)
        )
        self.layer = nn.Sequential(
            common.ResBlock(common.default_conv, features, 3, bn=True),
            nn.Conv2d(features, 4*features, 3, stride=2, padding=1),
            nn.BatchNorm2d(features*4), # 256
            nn.ReLU(inplace=True),

            common.ResBlock(common.default_conv, features*4, 3, bn=True),
            nn.Conv2d(4*features, 16*features, 3, stride=2, padding=1),
            nn.BatchNorm2d(features*16),
            nn.ReLU(inplace=True),

            common.ResBlock(common.default_conv, 16*features, 3, bn=True),
            nn.Conv2d(16*features, 32*features, 3, stride=2, padding=1),
            nn.BatchNorm2d(features*32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32*features, 32*features, 3, stride=2, padding=1),
            nn.BatchNorm2d(features*32),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

        self.tail = nn.Sequential(
            # nn.Conv2d(features, out_channel, 1),
            nn.Linear(in_features=32*features, out_features=1024, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.head(x)
        out = torch.squeeze(self.layer(out))
        # print(out.shape)
        out = self.tail(out)
        B,C = out.shape
        return out.view(B, -1)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


