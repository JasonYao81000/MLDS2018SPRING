import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import random
import matplotlib.pyplot as plt

random.seed(9487)

x = torch.unsqueeze(torch.linspace(0.0001, 1.0, 5000), dim=1)
y = torch.from_numpy(np.sinc(5*x))

epochs = 200
gradient_list = []
loss_list = []

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

training_loss = np.load('loss.npy')
minimum_ratio = []
x = Variable(x).cuda()
y = Variable(y).cuda()

for model in range(100):
	# print('epoch = {}'.format(e))
	net = Net(1, 1)
	net.cuda()
	net.load_state_dict(torch.load('./net_params_{}.pkl'.format(model)))

	loss_func = torch.nn.MSELoss()
	optimal_loss = training_loss[model]

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