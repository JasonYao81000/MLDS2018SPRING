import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import random
import matplotlib.pyplot as plt

random.seed(0)

x = torch.unsqueeze(torch.linspace(0.0001, 1.0, 5000), dim=1)
y = torch.from_numpy(np.sinc(5*x))


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()

        self.hidden1 = torch.nn.Linear(n_feature, 5)
        self.hidden2 = torch.nn.Linear(5, 10)
        # self.hidden3 = torch.nn.Linear(10, 10)
        # self.hidden4 = torch.nn.Linear(10, 10)
        # self.hidden5 = torch.nn.Linear(10, 10)
        # self.hidden6 = torch.nn.Linear(10, 10)
        self.hidden7 = torch.nn.Linear(10, 5)
        self.predict = torch.nn.Linear(5, n_output)

    def forward(self, x):
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        # x = F.selu(self.hidden3(x))
        # x = F.selu(self.hidden4(x))
        # x = F.selu(self.hidden5(x))
        # x = F.selu(self.hidden6(x))
        x = F.selu(self.hidden7(x))
        x = self.predict(x)
        return x

x = Variable(x)
y = Variable(y)

net_s = Net(1, 1)
net_s.load_state_dict(torch.load('sinc_init.pkl'))
net_f = Net(1, 1)
net_f.load_state_dict(torch.load('sinc.pkl'))
loss_func = torch.nn.MSELoss()

loss_total = []
a = np.linspace(0,1,1000)
for n,alpha in enumerate(a):
    for i,(p1,p2) in enumerate(zip(net_s.parameters(),net_f.parameters())):
        p1.data = (1-alpha)*p1.data + alpha*p2.data
    prediction = net_s(x)
    loss = loss_func(prediction, y)
    loss = loss.cpu() / 5000
    loss_total.append(loss.data[0])
plt.plot(a,loss_total)
plt.xlabel("alpha")
plt.ylabel("loss")
plt.savefig("loss_change.png")