import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


x = torch.unsqueeze(torch.linspace(0.0001, 1.0, 5000), dim=1) # x be two dimension data.
y = torch.from_numpy(np.sinc(5*x)) #+ 0.2*torch.rand(x.size())

loss_list = []

def gradient_P_norm():
	grad_all = 0

	for p in net.parameters():
		grad = 0.0
		if p.grad is not None:
			grad = (p.grad.cpu().data.numpy() ** 2).sum()
		grad_all += grad

	grad_norm = grad_all ** 0.5

	# print(grad_norm)
	return grad_norm

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
		x = F.relu(self.hidden1(x))
		x = F.relu(self.hidden2(x))
		# x = F.selu(self.hidden3(x))
		# x = F.selu(self.hidden4(x))
		# x = F.selu(self.hidden5(x))
		# x = F.selu(self.hidden6(x))
		x = F.relu(self.hidden7(x))
		x = self.predict(x)
		return x


train_loader = Data.DataLoader(dataset=Data.TensorDataset(x, y), batch_size=1024, shuffle=True)
# show the network.



# for num in range(20):
num = 0
while num < 100:
	net = Net(1, 1)
	net.cuda()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
	optimizer_g = torch.optim.Adam(net.parameters(), lr=0.0001)
	# optimizer_g = torch.optim.SGD(net.parameters(), lr = 0.0000001, momentum=0)
	
	loss_func = torch.nn.MSELoss()

	epoch = 64	#num*20
	# print('start training for total epoch = {}'.format(epoch))

	for t in range(epoch):

		epoch_loss = 0
		# epcoh_grad = 0
		for step, (bx, by) in enumerate(train_loader):  
			b_x = Variable(bx).cuda()   # batch x
			b_y = Variable(by).cuda()   # batch y
			prediction = net(b_x)

			loss = loss_func(prediction, b_y)
			epoch_loss += loss.data.cpu().numpy() / len(train_loader)
			# initial gradient to zero.
			optimizer.zero_grad()
			# run backpropagation.
			loss.backward()
			# update parameters.
			optimizer.step()
			# epcoh_grad += gradient_P_norm()
		print('Model[%03d] epoch: %03d, loss: %.6f' %(num, t, epoch_loss), end='\r')
	# loss_list.append(epoch_loss / len(train_loader))

	print('')
	# print('\nModel[%03d] Start searching for zero gradient %(num)')
	epoch_grad = 1024
	for t_g in range(epoch_grad):
		epoch_loss_g = 0
		epcoh_grad_g = 0

		for step_g, (bx_g, by_g) in enumerate(train_loader): 
			b_x_g = Variable(bx_g).cuda()  # batch x
			b_y_g = Variable(by_g).cuda()  # batch y
			prediction_g = net(b_x_g)

			loss_g = loss_func(prediction_g, b_y_g)
			epoch_loss_g += loss_g.data.cpu().numpy() / len(train_loader)
			
			optimizer_g.zero_grad()
			grad_all = torch.autograd.grad(loss_g,net.parameters(),  create_graph=True)

			grad_norm = Variable(torch.zeros(1, 1)).cuda()
			for i, g in enumerate(grad_all):
				grad_norm += (g ** 2).sum()
			grad_norm = grad_norm ** 0.5

			grad_norm.backward()
			optimizer_g.step()
			epcoh_grad_g += grad_norm.cpu().data.numpy() / len(train_loader)

		print('gradient epoch: %03d, loss: %.6f, gradient norm: %.6f' %(t_g, epoch_loss_g, epcoh_grad_g), end='\r')
		if epcoh_grad_g < 5e-3:
			loss_list.append(epoch_loss_g)
			torch.save(net.state_dict(), './net_params_{}.pkl'.format(num))
			num = num + 1
			# test_loss = 0
			# for step, (bx, by) in enumerate(train_loader):  
			# 	b_x = Variable(bx).cuda()   # batch x
			# 	b_y = Variable(by).cuda()   # batch y
			# 	prediction = net(b_x)
			# 	loss_test = loss_func(prediction, b_y)
			# 	test_loss += loss_test.data.cpu().numpy()
			# loss_list.append(test_loss / len(train_loader))
			# print('gradient epoch: %03d, loss: %.6f, gradient norm: %.6f' %(t_g, test_loss / len(train_loader), grad_norm.cpu().data.numpy()), end='\r')
			break

	print ('')
	# print('\nSave model[%03d].' %(num))
			
print('training finish.')

 
loss = np.array(loss_list)
np.save('loss.npy', loss)
print('save loss.') 

