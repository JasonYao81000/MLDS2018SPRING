import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

sens = np.load('./CIFAR_stat/sensitivity_list.npy')
train_loss = np.load('./CIFAR_stat/train_loss_list.npy')
train_acc = np.load('./CIFAR_stat/train_acc_list.npy')
test_loss = np.load('./CIFAR_stat/test_loss_list.npy')
test_acc = np.load('./CIFAR_stat/test_acc_list.npy')
print(sens)

batch = []
for i in range(3,14):
	batch.append(2**i)
batch = np.array(batch)

fig, ax1 = plt.subplots()
plt.xscale('log')
ax1.plot(batch, train_loss, 'b--',label='training loss')
ax1.plot(batch, test_loss, 'b-',label='testing loss')
ax1.set_xlabel('batch size')
ax1.set_ylabel('cross entropy loss', color='b')
ax1.tick_params('y', colors='b')
plt.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(batch, sens, 'r-')
ax2.set_ylabel('Sensitivity', color='r')
ax2.tick_params('y', colors='r')
red_patch = mpatches.Patch(color='red', label='Sensitivity')
plt.legend(handles=[red_patch],loc=1)

fig.tight_layout()
plt.savefig('CIFAR_loss.png')
plt.show()

fig, ax1 = plt.subplots()
plt.xscale('log')
ax1.plot(batch, train_acc, 'b--',label='training accuracy')
ax1.plot(batch, test_acc, 'b-',label='testing accuracy')
ax1.set_xlabel('batch size')
ax1.set_ylabel('Accuracy(%)', color='b')
ax1.tick_params('y', colors='b')
plt.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(batch, sens, 'r-')
ax2.set_ylabel('Sensitivity', color='r')
ax2.tick_params('y', colors='r')
plt.legend(handles=[red_patch],loc=1)

fig.tight_layout()
plt.savefig('CIFAR_acc.png')
plt.show()