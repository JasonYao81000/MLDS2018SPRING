from keras.models import load_model
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

'''
total_weight = []

for event in range(8):

	print('deal with {}'.format(event))
	for i in range(10):

		model = load_model('model/MNIST_{}_{}.hdf5'.format(event,(i+1)*3))
		weight = model.get_weights()
		epoch_weight = []
		for i,val in enumerate(weight):
			epoch_weight += list(val.flatten())

		total_weight.append(epoch_weight)

total_weight = np.array(total_weight)
print(total_weight.shape)

# vis_data = TSNE(n_components=2,perplexity=10,verbose=1,n_iter=10000).fit_transform(total_weight)
pca = PCA(n_components=2)
vis_data_pca = pca.fit_transform(total_weight)
print(vis_data_pca.shape)
np.save('vis.npy',vis_data_pca)
'''

loss = []
color = []

for event in range(8):
	color += [event]*10
	with open('history/MNIST_History_{}.csv'.format(event), 'r') as f:
		f.readline()
		for i, line in enumerate(f):
			if i%3 == 2:
				data = line.split(',')
				loss.append(round(float(data[4]),2))
		f.close()


vis_data = np.load('vis.npy')
plt.scatter(vis_data[:,0],vis_data[:,1],c=color)
for i, l in enumerate(loss):
	plt.text(vis_data[i,0],vis_data[i,1],l)
# plt.show()
plt.savefig('./picture/vis_opt_pca.png')
