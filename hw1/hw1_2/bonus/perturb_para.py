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

epochs = 200

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

net = Net(1, 1)
net.load_state_dict(torch.load('sinc.pkl'))

randomness = np.linspace(-0.001, 0.001, num=50)
loss_func = torch.nn.MSELoss()
loss_total = []
for i, p_layer in enumerate(net.parameters()):
	print("Deal with layer {}".format(i))
	if i % 2 == 0:
		for j in range(p_layer.data.size()[0]):
			for k in range(p_layer.data.size()[1]):
				loss_para = []
				origin = p_layer.data[j][k]
				for l,data in enumerate(randomness):
					p_layer.data[j][k] = origin + data
					prediction = net(x)
					loss = loss_func(prediction, y)
					loss = loss.cpu() / 5000
					loss_para.append(loss.data[0])
				p_layer.data[j][k] = origin
				loss_total.append(loss_para)

loss_total = np.array(loss_total)
for i in range(len(loss_total)):
	plt.plot(randomness,loss_total[i])
plt.xlabel("offset")
plt.ylabel("loss")
plt.savefig("perturb.png")

'''
sample_num = 10000
minimum_num = 0
for sample in range(sample_num):
	net.load_state_dict(torch.load('./net_params_{}.pkl'.format(model)))

	for index, f in enumerate(net.parameters()):
		offset = (random.random() - 0.5) * 2
		f.data.add_(offset)

	prediction = net(x)
	loss = loss_func(prediction, y)
	loss = loss.cpu() / 5000
	if loss.data[0] > optimal_loss:
		minimum_num += 1
	if (sample % (sample_num / 100) == 0):
		print ('Minimum_ratio: %06d / %06d' %(minimum_num, sample), end='\r')

	minimum_ratio.append(minimum_num / sample_num)
	print ('\nModel[%03d] Minimum_ratio: %.6f, loss: %.6f' %(model, minimum_num / sample_num, training_loss[model]))
minimum_ratio = np.array(minimum_ratio)
np.save('minimum_ratio.npy',minimum_ratio)
'''