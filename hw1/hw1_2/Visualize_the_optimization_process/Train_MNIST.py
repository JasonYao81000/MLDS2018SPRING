import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from keras.utils import np_utils

from keras.datasets import mnist
from keras.models import Model,load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization, Convolution2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model():

	model=Sequential()
	
	model.add(Dense(units=128, input_dim=784, activation='relu'))
	model.add(Dense(units=128, activation='relu'))  
	model.add(Dense(units=10, activation='softmax')) 

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def _shuffle(X, Y):

	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])

def Train(model,X_train,Y_train,outdir):
	
	callbacks = [
		# EarlyStopping(monitor='val_acc', min_delta=1e-4,
		# patience = 10, verbose = 0, mode='max'),
		ModelCheckpoint(outdir, verbose = 0, save_weights_only=False, period=3)]
	
	His = model.fit(X_train, Y_train, batch_size=64, verbose = 1,validation_split=0.1,
		epochs=30, callbacks=callbacks, shuffle=True)
	
	return His

(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  
x_Train = X_train_image.reshape(60000, 28*28).astype('float32')  
x_Test = X_test_image.reshape(10000, 28*28).astype('float32')  
  
# Normalization  
x_Train_norm = x_Train/255  
x_Test_norm = x_Test/255  

y_TrainOneHot = np_utils.to_categorical(y_train_label) 
y_TestOneHot = np_utils.to_categorical(y_test_label)

[x_Train_norm,y_TrainOneHot] = _shuffle(x_Train_norm,y_TrainOneHot)
model = build_model()
His = Train(model,x_Train_norm,y_TrainOneHot,'model/MNIST_0_{epoch}.hdf5')
pd.DataFrame(His.history).to_csv("history/MNIST_History_0.csv")

