import argparse
import os
import shutil
import time
import importlib

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image, ImageOps
import numpy as np

import TCL as TCL
from utils import *
import caffe_transform as caffe_t
from data import ImageList

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ACAN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=31, type=int, metavar='N',
                    help='number of classes (default: 31)')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck (default: 256)')
parser.add_argument('--gpu', default='0', type=str, metavar='N',
                    help='visible gpu')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', default=10.0, type=float, metavar='M',
                    help='dloss weight')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--train-iter', default=50000, type=int,
                    metavar='N', help='')
parser.add_argument('--test-iter', default=300, type=int,
                    metavar='N', help='')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--alpha', default=10.0, type=float, metavar='M')
parser.add_argument('--beta', default=0.75, type=float, metavar='M')
parser.add_argument('-hl', '--hidden', default=1024, type=int, metavar='N',
                    help='width of hiddenlayer (default: 1024)')
parser.add_argument('--name', default='alexnet', type=str)

parser.add_argument('--dataset', default='None', type=str)
parser.add_argument('--traindata', default='None', type=str)
parser.add_argument('--valdata', default='None', type=str)

parser.add_argument('--noiselevel', default='None', type=str)
parser.add_argument('--noisetype', default='None', type=str)

parser.add_argument('--traded', default=1.0, type=float)
parser.add_argument('--tradet', default=1.0, type=float)

parser.add_argument('--startiter', default=3000, type=int)
parser.add_argument('--Lythred', default=0.5, type=float)
parser.add_argument('--Ldthred', default=0.5, type=float)
parser.add_argument('--lambdad',default=1.0,type=float)


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # create model
    net = TCL.TCL_Net(args).cuda()


    # Data loading code
    if args.noisetype == 'corruption':
        if args.dataset == 'officehome':
            traindir = './officehome_corrupted_list/'+args.traindata+'_corrupted_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'
        elif args.dataset == 'office':
            traindir = './office_corrupted_list/'+args.traindata+'_list_corrupted_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'

    elif args.noisetype == 'noise':
        if args.dataset == 'officehome':
            traindir = './officehome_list/'+args.traindata+'_noisy_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'

        elif args.dataset == 'office':
            traindir = './office_list/'+args.traindata+'_list_noisy_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'
    
    elif args.noisetype == 'both':
        if args.dataset == 'officehome':
            traindir = './officehome_noisycorrupted_list/'+args.traindata+'_noisycorrupted_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'
        elif args.dataset == 'office':
            traindir = './office_noisycorrupted_list/'+args.traindata+'_list_noisycorrupted_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'



    #set data_transforms
    data_transforms = {
      'train': caffe_t.transform_train(resize_size=256, crop_size=224),
      'val': caffe_t.transform_train(resize_size=256, crop_size=224),
  }
    data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)

 
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(traindir).readlines(), 
        transform = data_transforms["train"]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    target_loader = torch.utils.data.DataLoader(
        ImageList(open(valdir).readlines(), 
        transform = data_transforms["val"]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        ImageList(open(valdir).readlines(), 
        transform = data_transforms["val9"]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    TCL.train_val(source_loader, target_loader, val_loader,
                     net, args)


if __name__ == '__main__':
    main()
