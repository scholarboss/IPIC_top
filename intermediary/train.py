from utils import *
import struct
from model import *
from dataloader import *
import torch
import numpy as np
import argparse
import h5py
from transformer import *
from data_parallel import BalancedDataParallel
# from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1.0e-03, help='learning rate')
parser.add_argument('--decay', type=int, default=50, help='learning rate decay')
parser.add_argument('--start', type=int, default=0, help='start epoch')
parser.add_argument('--repeat_begin', type=int, default=1, help='start epoch')
parser.add_argument('--repeat_end', type=int, default=1, help='start epoch')
parser.add_argument('--end', type=int, default=3000, help='end epoch')
parser.add_argument('--pilot', type=int, default=8, help='8 or 32')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--bn', type=bool, default=False, help='BatchNorm')
parser.add_argument('--channel', type=int, default=64, help='model channel')
parser.add_argument('--freq', type=int, default=10, help='print frequency')
parser.add_argument('--num_res', type=int, default=1, help='resblock num')
parser.add_argument('--embed', type=int, default=2048, help='embed dim')
parser.add_argument('--tag', type=str, default='', help='tag on checkpoint')
parser.add_argument('--ckp', type=str, default='none', help='checkpoint name')
parser.add_argument('--optim', type=str, default='Adam', help='Adam or SGD')
parser.add_argument('--loss', type=str, default='2', help='1 denote mse, 2 denote CrossEntropy ')
parser.add_argument('--model', type=str, default='2', help='1: , 2:Linear, 3:transformer 4:Linear_deep 5:res 6:res deep')
parser.add_argument('--nhid', type=int, default=512, help='transformer hidden unit')
parser.add_argument('--nlayers', type=int, default=16, help='transformer layers num')

# nhid=512, nlayers=16

args = parser.parse_args()

if args.model == '1':
    model = CNNModel(features=args.channel).cuda()
elif args.model == '2':
    model = TextSentiment(num_class=1024+args.pilot*4).cuda()
elif args.model == '3':
    model = TransformerModel(nhid=args.nhid, nlayers=args.nlayers).cuda()
elif args.model == '4':
    model = TextSentiment_deep(num_class=1024+args.pilot*4).cuda()
elif args.model == '5': # res
    model = TextSentiment_res(num_class=1024+args.pilot*4).cuda()
elif args.model == '6': # res deep
    model = DnText_res(embed_dim=args.embed ,num_class=1024+args.pilot*4).cuda()
elif args.model == '7': # res deep
    model = DnText_linear(num_class=1024+args.pilot*4, bn=args.bn).cuda()
elif args.model == '8': # res deep
    model = UNet(embed_dim=args.embed, num_class=1024+args.pilot*4, num_res=args.num_res, bn=args.bn).cuda()
elif args.model == '9': # res deep
    model = UNetAttention(embed_dim=args.embed, num_class=1024+args.pilot*4, num_res=args.num_res, bn=args.bn).cuda()
elif args.model == '10': # res deep
    model = UNetAttention_v2(embed_dim=args.embed, num_class=1024+args.pilot*4, num_res=args.num_res, bn=args.bn).cuda()
    # model = BalancedDataParallel(args.batchsize//4, model).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # model.head.to('cuda:0')
    # model.layer_i1.to('cuda:0')
    # model.layer_i2.to('cuda:0')
    # model.layer_i3.to('cuda:0')
    # model.layer_i4.to('cuda:0')
    # model.layer_i5.to('cuda:0')
    # model.layer_o5.to('cuda:1')
    # model.layer_o4.to('cuda:1')
    # model.layer_o3.to('cuda:1')
    # model.layer_o2.to('cuda:1')
    # model.tail.to('cuda:1')



print_network(model)

if args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == 'SGD':
    # lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_score = 0.0
if args.ckp != 'none':
    encoder_path = './Modelsave/{}.pth'.format(args.ckp)
    state = torch.load(encoder_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer_state_dict'])
    best_score = state['score']
    print('load checkpoint success!!!')


data1=open('data/H.bin','rb')
data1.seek(4*2*2*2*32*300000)

H1=struct.unpack('f'*2*2*2*32*20000,data1.read(4*2*2*2*32*20000))
H1=np.reshape(H1,[20000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]
Htest=H[0:,:,:]


# H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
# H1=np.reshape(H1,[320000,2,4,32])
# H=H1[:,1,:,:]+1j*H1[:,0,:,:]

# H=H[:1,:,:]
# H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
# H1=np.reshape(H1,[320000,2,4,32])
# H=H1[:,1,:,:]+1j*H1[:,0,:,:]

# Htest=H[300000:,:,:]
# H=H[:300000,:,:]

# H1=struct.unpack('f'*2*2*2*32*10000,data1.read(4*2*2*2*32*10000))
# H1=np.reshape(H1,[10000,2,4,32])
# H=H1[:,1,:,:]+1j*H1[:,0,:,:]

# Htest=H[8000:,:,:]
# H=H[:8000,:,:]

learning_rate = args.lr



# train_data = ChannelDataLoader(H, pilot=args.pilot)
# train_data = LmdbDataLoader(pilot=args.pilot, repeat=args.repeat)
# train_data = NpyDataLoader(pilot=args.pilot, num=0)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
test_data = ChannelDataLoader(Htest, pilot=args.pilot)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batchsize, shuffle=True, num_workers=6)

if args.loss == '1':
    criterion = torch.nn.MSELoss().cuda()
elif args.loss == '2':
    criterion = torch.nn.BCELoss().cuda()





train_data = None
input_data = None
label_data = None
for epoch in range(args.start, args.end):
    
    # choose different datalset
    num = np.random.randint(args.repeat_begin, args.repeat_end)
    # dir = 'data/NumPy_Pilot_{}/Data_{}.npz'.format(args.pilot, num)
    # data = np.load(dir)
    # train_data = NpyDataLoader(data=data, pilot=args.pilot)
    # data = h5py.File('data/H5py_Pilot_{}/Data_{}.hdf5'.format(args.pilot, num), 'r')
    
    '''H5py dataset'''
    # train_data = H5pyDataLoader(pilot=args.pilot, num=num)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    
    '''Bin dataset'''
    del input_data
    del label_data
    input_name = 'data/BIN_Pilot_{}/input_{}.bin'.format(args.pilot, num)
    f_input = open(input_name, 'rb')
    label_name = 'data/BIN_Pilot_{}/label_{}.bin'.format(args.pilot, num)
    f_label = open(label_name, 'rb')
    print('load train data {} begin...'.format(input_name))
    input_data = f_input.read(4*2048*300000)
    # input_data = np.reshape(input_data, [300000, 2048])
    label_data = f_label.read(1024*300000)
    # label_data = np.reshape(label_data, [300000, 1024])    
    # input_data = struct.unpack('f'*2048*300000, f_input.read(4*2048*300000))
    # input_data = np.reshape(input_data, [300000, 2048])
    # label_data = struct.unpack('?'*1024*300000, f_label.read(1024*300000))
    # label_data = np.reshape(label_data, [300000, 1024])

    train_data = BinDataLoader(X=input_data, Y=label_data, pilot=args.pilot)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=2)#, pin_memory=True)
    print('load train data successful...')



    model.train()
    if epoch % args.decay == 0 and epoch != 0:
        print('learning rate decay from {:.8f} to {:.8f}'.format(learning_rate, learning_rate*0.1))
        learning_rate = learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    # measure data loading time    

    for i, (input_, label_) in enumerate(train_loader):
        if args.model == '1':
            # input_, label_ = input_.cuda(), label_.cuda()
            B,N = input_.shape
            input_ = torch.cat((input_[:,::2].view(B,1,32,32), input_[:,1::2].view(B,1,32,32)), 1)
            # input_ = input_.view(B, 2, 32, 32)
        input_, label_ = input_.cuda(), label_.cuda()      
        out = model(input_)
        loss = criterion(out, label_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.freq == 0:
            print('Train:> Epoch: [{0}][{1}/{2}]\t'
                  'lr {lr:.8f} | Loss {loss:.6f}\t'.format(
                epoch, i+1, len(train_loader), lr=learning_rate, loss=loss.item()))

    model.eval()
    total_score = 0
    
    with torch.no_grad():
        for i, (input_, label_) in enumerate(test_loader):
            # convert numpy to Tensor
            input_, label_ = input_.cuda(), label_.cuda()  
            if args.model == '1':
                B,N = input_.shape
                input_ = input_.view(B, 2, 32, 32)
            out = model(input_)
            # out[out<0.5] = 0
            # out[out>=0.5] = 1
            out = torch.round(out)
            error = torch.sum(torch.abs(label_[-1024:]-out[-1024:])).item()
            # print()
            B, _ = label_.shape
            score = 1 - error/(B*1024)
            if np.isnan(score):
                print('test:> step {} | batchsize: {} | error: {} | score: {:.6f} | score-bef: {}'.format(i, B, error, score, total_score/(i+1)))
                break
            total_score += score
        print('Test:> score: {} ++++++++++.++++++++++'.format(total_score/len(test_loader)))
        if best_score < total_score/len(test_loader):
            best_score = total_score/len(test_loader)
            modelSave = './Modelsave/model_pilot{}_{}.pth'.format(args.pilot, args.tag)
            state = {
                'state_dict': model.state_dict(),
                # 'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'score': best_score
                }
            torch.save(state, modelSave)
            print('model saved to {}. score is {}'.format(modelSave, best_score))





