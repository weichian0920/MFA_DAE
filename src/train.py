#/--coding:utf-8/
#/author:Ethan Wang/

import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import misc, make_path, signalprocess



def train(train_loader, net=None, args=None, logger=None):
    best_acc, old_file = 0, None
    per_save_epoch = 30
    #optimizer
    if args.optim=="RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optim=="Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    #loss function
    mse = nn.MSELoss()
    train_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)

    best_loss = 100
    old_file = 0
    #start training
    for epoch in range(args.epochs):
        avg_batch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data)
            optimizer.zero_grad()
            output = net(data)  
            loss = mse(output, data)
            avg_batch_loss +=loss
            loss.backward()
            optimizer.step()
            #train_scheduler.step(epoch+batch_idx/311)

            ##validation
            """
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                if loss>best_loss:
                    new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                    misc.model_save(model, new_file, old_file=old_file, verbose=True)
                    best_loss = loss
                    old_file = new_file
            """
        new_file = os.path.join(args.logdir, 'latest.pth')
        misc.model_save(net, new_file, old_file=old_file, verbose=True)
        old_file = new_file
        
        logger("epoch{0}:{1}".format(epoch, avg_batch_loss/(batch_idx)))

if __name__=="__main__":
    train_test()
