from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.selu(F.max_pool2d(self.conv1(x), 2))
        x = F.selu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.selu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(epoch,train_loader,model):
    sens = 0
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data,requires_grad=True), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]),end='\r')
        if epoch == args.epochs:    
            sens_batch = torch.norm(data.grad.data.view(len(data),-1),2,1).mean()
            sens += sens_batch
    loss,acc = test(train_loader,model)
    sens = sens/len(train_loader.dataset)
    return loss,acc,sens

def test(test_loader,model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return test_loss,acc

sensitivity_list = []
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(3,14):

    model = Net()
    if args.cuda:
        model.cuda()

    batch = 2**i
    print('Begin training on batch size = {}'.format(batch))
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1024, shuffle=True, **kwargs)

    trn_loss = 0;trn_acc = 0;s = 0;test_loss = 0;test_acc = 0;
    for epoch in range(1, args.epochs + 1):
        x,y,z = train(epoch,train_loader,model)
        w,k = test(test_loader,model)
        if k > test_acc:
        	trn_loss = x;trn_acc = y;s = z;test_loss = w;test_acc = k

    print('Sensitivity = {}'.format(z))
    sensitivity_list.append(z)
    train_loss_list.append(x)
    test_loss_list.append(w)
    train_acc_list.append(y)
    test_acc_list.append(k)

sensitivity_list = np.array(sensitivity_list)
np.save('sensitivity_list.npy', sensitivity_list)
train_loss_list = np.array(train_loss_list)
np.save('train_loss_list.npy', train_loss_list)
test_loss_list = np.array(test_loss_list)
np.save('test_loss_list.npy', test_loss_list)
train_acc_list = np.array(train_acc_list)
np.save('train_acc_list.npy', train_acc_list)
test_acc_list = np.array(test_acc_list)
np.save('test_acc_list.npy', test_acc_list)
