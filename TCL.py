import argparse
import pdb
import os
import shutil
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import math
from utils import *
import numpy as np

class GRL_Layer(torch.autograd.Function):
    def __init__(self, args):
        self.gamma = args.gamma
        self.total_iter = 10000.0

    def forward(self, input):
        output = input * 1.0
        return output

    def backward(self, grad_output):
        global global_iter
        lamda = 2.0 / (1.0 + math.exp(- self.gamma * global_iter / self.total_iter)) -1
        return (- lamda) * grad_output


class TCL_Net(nn.Module):
    def __init__(self, args):
        super(TCL_Net, self).__init__()
        #create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating new model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=False)

        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            self.feature_dim = model.classifier[6].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        else:
            self.feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-1])

        #Feature extractor and Classifier
        self.feature = model
        self.bottleneck = nn.Linear(self.feature_dim, args.bottleneck)
        self.classifier = nn.Linear(args.bottleneck, args.classes)
        
        for param in self.feature.parameters():
            param.requires_grad = False
        
        #Domain Discriminator
         
        self.relu = nn.ReLU()
        self.dfc1 = nn.Linear(args.bottleneck, args.hidden)
        self.dfc2 = nn.Linear(args.hidden, args.hidden)
        self.discriminator = nn.Linear(args.hidden, 1)
        self.grld = GRL_Layer(args)
        self.sigmoid = nn.Sigmoid()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        #initialize
         
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.dfc1.weight.data.normal_(0, 0.01)
        self.dfc1.bias.data.fill_(0.0)
        self.dfc2.weight.data.normal_(0, 0.01)
        self.dfc2.bias.data.fill_(0.0)
        self.discriminator.weight.data.normal_(0, 0.3)
        self.discriminator.bias.data.fill_(0.0)
        self.softmax = nn.Softmax()

    def forward(self, x):
        #feature extractor
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)

        #classifier
        y = self.classifier(x)
        y_softmax = self.softmax(y)
        
        #domain discriminator
        xd = self.grld(x)

        #xd with grl 
        xd = self.drop1(self.relu(self.dfc1(xd)))
        xd = self.drop2(self.relu(self.dfc2(xd)))
        d = self.discriminator(xd)
        d = self.sigmoid(d)
 
        return y , y_softmax , d

def train_val(source_loader, target_loader, val_loader, net, args):
    net.train()
    net.feature.eval()
    
    #optimizer
    sgd_param = [
            {'params': net.bottleneck.parameters(), 'lr':1},
            {'params': net.classifier.parameters(), 'lr':1},            
            {'params': net.dfc1.parameters(), 'lr':1},             
            {'params': net.dfc2.parameters(), 'lr':1},
            {'params': net.discriminator.parameters(), 'lr':1},    
        ]  
    sgd_weight_param = [1,1,1,1,1]
    
    optimizer = torch.optim.SGD(sgd_param, args.lr, 
                                momentum=args.momentum,
                                weight_decay = args.weight_decay)
    
    source_domain_label = torch.FloatTensor(args.batch_size,1)
    target_domain_label = torch.FloatTensor(args.batch_size,1)
    source_domain_label.fill_(1)
    target_domain_label.fill_(0)
    domain_label = torch.cat([source_domain_label,target_domain_label],0)
    domain_label = torch.autograd.Variable(domain_label.cuda()) 

    #training info meter
    batch_time_Meter = AverageMeter()
        #Accuracy
    top1_Meter = AverageMeter()
        #Loss
    lossY_Meter = AverageMeter()
    lossD_Meter = AverageMeter()
    lossT_Meter = AverageMeter()
    sourceN_Meter = AverageMeter()

    #--------
    source_len = len(source_loader) - 1
    target_len = len(target_loader) - 1

    #loss criterion
    label_criterion = nn.CrossEntropyLoss().cuda()
    domain_criterion = nn.BCELoss().cuda()

    endtime = time.time()
    best_prec1 = 0

    #create save directory
    savepath = "runs/TCL/%s/"%(args.name)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_out = open(savepath + 'train_log.txt','w')

    #train train_iters
    for i in range(args.train_iter):
        global global_iter
        global_iter = i
        optimizer = adjust_learning_rate(optimizer, i, args, sgd_weight_param)

        if i % source_len == 0:
            iter_source = iter(source_loader)
        if i % target_len == 0:
            iter_target = iter(target_loader)
        source_input, label = iter_source.next()
        target_input, _ = iter_target.next()
        
        inputs = torch.cat((source_input, target_input),0)
        input_var = torch.autograd.Variable(inputs.cuda())
        label_var = torch.autograd.Variable(label.cuda())
        label = label.cuda()
        y_var, y_softmax_var, d_var = net(input_var)
        source_y, target_y = y_var.chunk(2,0)
        source_y_softmax, target_y_softmax = y_softmax_var.chunk(2,0)
        source_d, target_d = d_var.chunk(2,0)
      
        #calculate Ly 
        if i < args.startiter:
            Ly = label_criterion(source_y, label_var)
        else:
            Ly, source_weight, source_num = cal_Ly(source_y_softmax, source_d, label, args)
            sourceN_Meter.update(source_num)

            target_weight = torch.ones(source_weight.size()).cuda()

        #calculate Lt
        Lt = cal_Lt(target_y_softmax)

        #calculate Ld
        if i < args.startiter:
            Ld = domain_criterion(d_var, domain_label)
        else:
            domain_weight = torch.cat([source_weight,target_weight],0)
            domain_weight = domain_weight.view(-1,1)
            domain_criterion = nn.BCELoss(weight=domain_weight).cuda()
            Ld = domain_criterion(d_var, domain_label)

        loss = Ly + args.traded * Ld + args.tradet * Lt
        
        optimizer.zero_grad()
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Training infomation
        prec1, _ = accuracy(source_y.data, label_var.data, topk=(1,5))
        batch_time_Meter.update(time.time() - endtime)
        lossY_Meter.update(Ly.data[0])
        lossD_Meter.update(Ld.data[0])
        lossT_Meter.update(Lt.data[0])
        top1_Meter.update(prec1[0])
        endtime = time.time()

    # Print
        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]  '
            'Time: {batch_time_Meter.avg:.3f}  '
            'Ly: {lossY_Meter.avg:.3f}  '
            'Ld: {lossD_Meter.avg:.3f}  '
            'Lt: {lossT_Meter.avg:.3f}  '
            'sourceN: {sourceN_Meter.avg:.2f}  '
            'Prec1: {top1_Meter.avg:.2f}'.format(
            i, args.train_iter, batch_time_Meter = batch_time_Meter,
            lossY_Meter = lossY_Meter, lossD_Meter = lossD_Meter,
            lossT_Meter = lossT_Meter,
            sourceN_Meter=sourceN_Meter, 
            top1_Meter = top1_Meter))        

            file_out.write('Iter: [{0}/{1}]  '
            'Time: {batch_time_Meter.avg:.3f}  '
            'Ly: {lossY_Meter.avg:.3f}  '
            'Ld: {lossD_Meter.avg:.3f}  '
            'Lt: {lossT_Meter.avg:.3f}  '
            'sourceN: {sourceN_Meter.avg:.2f}  '
            'Prec1: {top1_Meter.avg:.2f}\n'.format(
            i, args.train_iter, batch_time_Meter = batch_time_Meter,
            lossY_Meter = lossY_Meter, lossD_Meter = lossD_Meter,
            lossT_Meter = lossT_Meter,
            sourceN_Meter=sourceN_Meter,             
            top1_Meter = top1_Meter))        

            file_out.flush()

        if i % args.test_iter ==0 and i != 0:
            val_prec1 = validate(val_loader, net, args, file_out)
            batch_time_Meter.reset()
            lossY_Meter.reset()
            lossD_Meter.reset()
            lossT_Meter.reset()
            top1_Meter.reset()
            sourceN_Meter.reset()
            if best_prec1 < val_prec1:
                best_prec1 = val_prec1
                #torch.save(net.state_dict(), savepath+'net_best.pth')
    print('Best Prec1: %.2f' % (best_prec1))    
    file_out.write('Best Prec1: %.2f\n' % (best_prec1)) 
    file_out.flush()
    file_out.close()
    #torch.save(net.state_dict(), savepath+'net_final.pth')


def validate(val_loader, netG, args, file_out):
    top1_Meter = AverageMeter()
    netG.eval()
    for i, (val_input,val_label) in enumerate(val_loader):
        input_var = torch.autograd.Variable(val_input, volatile=True).cuda()
        label_var = torch.autograd.Variable(val_label, volatile=True).cuda()

        output , _ ,  _  = netG(input_var)
        prec1, _ = accuracy(output.data, label_var.data, topk=(1,5))
        top1_Meter.update(prec1[0],val_input.size(0))
        
    print('* Validation Prec1 {top1_Meter.avg:.3f}'.format(
        top1_Meter=top1_Meter))
    file_out.write('* Validation Prec1 {top1_Meter.avg:.3f}\n'.format(
        top1_Meter=top1_Meter))
    file_out.flush()

    netG.train()
    netG.feature.eval()
    return top1_Meter.avg

def cal_Ly(source_y_softmax, source_d, label, args):
    agey = - math.log(args.Lythred)
    aged = - math.log(1.0 - args.Ldthred)
    age = agey + args.lambdad * aged
    y_softmax = source_y_softmax
    the_index = torch.LongTensor(np.array(range(args.batch_size))).cuda()
    y_label = y_softmax[the_index, label]
    y_loss = - torch.log(y_label)

    d_loss = - torch.log(1.0 - source_d)
    d_loss = d_loss.view(args.batch_size)

    weight_loss = y_loss + args.lambdad * d_loss

    weight_var = (weight_loss < age).float().detach()
    Ly = torch.mean(y_loss * weight_var)

    source_weight = weight_var.data.clone()
    source_num = float((torch.sum(source_weight)))
    return Ly, source_weight, source_num


def cal_Lt(target_y_softmax):
    Gt_var = target_y_softmax
    Gt_en = - torch.sum((Gt_var * torch.log(Gt_var + 1e-8)), 1)
    Lt = torch.mean(Gt_en)
    return Lt






