'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from vgg import *
from utils import progress_bar
from torch.autograd import Variable
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, metavar='N',help='number of epochs to train (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy


criterion = nn.CrossEntropyLoss()


# Training
def train(epoch,trainloader,net):

    print('\nEpoch: %d' % epoch)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    train_loss = 0
    correct = 0
    total = 0
    sens = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs,requires_grad=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if epoch == args.epochs:    
            sens_batch = torch.norm(inputs.grad.data.view(len(inputs),-1),2,1).mean()
            sens += sens_batch
        sens = sens/len(trainloader.dataset)

        loss = train_loss/(batch_idx+1)
        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (loss, acc, correct, total))
        

    return loss,acc,sens


def test(epoch,testloader,net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss = test_loss/(batch_idx+1)
        acc = 100.*correct/total
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (loss, acc, correct, total))

    
    return loss,acc

    
sensitivity_list = []
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(3,11):

    batch = 2**i
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = VGG('VGG11')

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    print('Begin training on batch size = {}'.format(batch))
    for epoch in range(1,args.epochs+1):
        x,y,z = train(epoch,trainloader,net)
        w,k = test(epoch,testloader,net)

    print('Sensitivity = {}'.format(z))
    sensitivity_list.append(z)
    train_loss_list.append(x)
    test_loss_list.append(w)
    train_acc_list.append(y)
    test_acc_list.append(k)

sensitivity_list = np.array(sensitivity_list)
np.save('./CIFAR_stat/sensitivity_list.npy', sensitivity_list)
train_loss_list = np.array(train_loss_list)
np.save('./CIFAR_stat/train_loss_list.npy', train_loss_list)
test_loss_list = np.array(test_loss_list)
np.save('./CIFAR_stat/test_loss_list.npy', test_loss_list)
train_acc_list = np.array(train_acc_list)
np.save('./CIFAR_stat/train_acc_list.npy', train_acc_list)
test_acc_list = np.array(test_acc_list)
np.save('./CIFAR_stat/test_acc_list.npy', test_acc_list)
