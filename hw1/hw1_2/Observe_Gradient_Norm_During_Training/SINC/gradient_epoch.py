import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


x = torch.unsqueeze(torch.linspace(0.0001, 1.0, 5000), dim=1) # x be two dimension data.
y = torch.from_numpy(np.sinc(5*x)) #+ 0.2*torch.rand(x.size())

# x, y = Variable(x), Variable(y)
epochs = 700
gradient_list = []
loss_list = []

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
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
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        # x = F.selu(self.hidden3(x))
        # x = F.selu(self.hidden4(x))
        # x = F.selu(self.hidden5(x))
        # x = F.selu(self.hidden6(x))
        x = F.selu(self.hidden7(x))
        x = self.predict(x)
        return x

net = Net(1, 1)
train_loader = Data.DataLoader(dataset=Data.TensorDataset(x, y), batch_size=128, shuffle=True)
# show the network.
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

for t in range(epochs):
    epoch_loss = 0
    epcoh_grad = 0
    for step, (bx, by) in enumerate(train_loader):  
        b_x = Variable(bx)   # batch x
        b_y = Variable(by)   # batch y
        prediction = net(b_x)

        loss = loss_func(prediction, b_y)
        epoch_loss += loss.data.numpy()
        # initial gradient to zero.
        optimizer.zero_grad()
        # run backpropagation.
        loss.backward()
        # update parameters.
        optimizer.step()
        epcoh_grad += gradient_P_norm()

    # print(loss.shape)
    print('epoch: %d, loss: %.6f' %(t, epoch_loss / len(train_loader)))
    
    gradient_list.append(epcoh_grad / len(train_loader))
    loss_list.append(epoch_loss / len(train_loader))


print('training finish.')

torch.save(net, 'sinc.pkl') 
print('save model.') 
gradient = np.array(gradient_list)
np.save('gradientNorm.npy', gradient)
loss = np.array(loss_list)
np.save('loss.npy', loss)
print('save loss.') 

