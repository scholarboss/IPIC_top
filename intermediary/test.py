from utils import *
import struct
from model import *
from dataloader import ChannelDataLoader
import torch
import argparse
from transformer import TextSentiment
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--channel', type=int, default=128, help='model channel')
parser.add_argument('--tag', type=str, default='', help='tag on checkpoint')
parser.add_argument('--ckp', type=str, default='none', help='checkpoint name')
parser.add_argument('--model', type=str, default='2', help='1: , 2: ')
parser.add_argument('--data', type=str, default='Y_1', help='Y_1 used pilot 32 , Y_2 8')



args = parser.parse_args()

if args.model == '1':
    model = CNNModel(features=args.channel).cuda()
elif args.model == '2':
    model = TextSentiment().cuda()
print_network(model)

if args.ckp != 'none':
    encoder_path = './Modelsave/{}.pth.tar'.format(args.ckp)
    model.load_state_dict(torch.load(encoder_path)['state_dict'])
    print('load checkpoint success!!!')


Y = open('data/{}.csv'.format(args.data), 'r')
YY = Y.readlines()
model_out = []
for i in tqdm(range(len(YY))):
    line = YY[i].split(',')
    line = np.array(line, dtype=np.float)
    line = torch.tensor(line).float().cuda()
    out = model(line)
    out = out.cpu().detach().numpy()
    model_out.append(out)

model_out = np.asarray(model_out)
model_out = np.array(np.floor(model_out + 0.5), dtype=np.bool)
model_out.tofile('data/X_pre_{}.bin'.format(args.data[-1]))





