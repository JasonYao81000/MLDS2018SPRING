# python3 interpolation.py 

import sys
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model
from keras.datasets import mnist

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')
    
    # Input image dimensions.
    rowsImage, colsImage = 28, 28

    # the data, split between train and test sets
    (x_data_train, y_data_train), (x_data_test, y_data_test) = mnist.load_data()
    
    # Reshape (rowsImage * colsImage) to (rowsImage, colsImage) array.
    if K.image_data_format() == 'channels_first':
        x_data_train = x_data_train.reshape((x_data_train.shape[0], 1, rowsImage, colsImage))
        x_data_test = x_data_test.reshape(x_data_test.shape[0], 1, rowsImage, colsImage)
        input_shape = (1, rowsImage, colsImage)
    else:
        x_data_train = x_data_train.reshape((x_data_train.shape[0], rowsImage, colsImage, 1))
        x_data_test = x_data_test.reshape(x_data_test.shape[0], rowsImage, colsImage, 1)
        input_shape = (rowsImage, colsImage, 1)
    
    # Scale to 0 ~ 1 on training set.
    x_data_train = x_data_train.astype('float32') / 255.0
    x_data_test = x_data_test.astype('float32') / 255.0
    print ('Scaled to 0 ~ 1 on training set.')

    # Convert labels to one hot encoding.
    y_data_train = to_categorical(y_data_train)
    y_data_test = to_categorical(y_data_test)

    # Load models from h5 file.
    model_64 = load_model('MNIST_1e-3.h5')
    model_1024 = load_model('MNIST_1e-2.h5')
    model_interpolation = load_model('MNIST_1e-3.h5')

    # Get weights from model and convert to np.array.
    weights_64 = np.array(model_64.get_weights())
    weights_1024 = np.array(model_1024.get_weights())

    # Interpolation ratio.
    alphas = np.linspace(start=-1, stop=2, num=50)
    # Loss and Acc. in training and testing set.
    train_loss = np.zeros(alphas.shape)
    train_acc = np.zeros(alphas.shape)
    test_loss = np.zeros(alphas.shape)
    test_acc = np.zeros(alphas.shape)

    for index, alpha in enumerate(alphas):
        # Linear interpolation with ratio alpha.
        weights_interpolation = (1 - alpha) * weights_64 + alpha * weights_1024
        # Set interpolation weights to model.
        model_interpolation.set_weights(list(weights_interpolation))
        # Evaluate loss and Acc in training and testing set.
        train_loss[index], train_acc[index] = model_interpolation.evaluate(x_data_train, y_data_train, batch_size=1024, verbose=0)
        test_loss[index], test_acc[index] = model_interpolation.evaluate(x_data_test, y_data_test, batch_size=1024, verbose=0)
        # Save loss and Acc. in training and testing set to npy file.
        np.save('train_loss.npy', train_loss)
        np.save('train_acc.npy', train_acc)
        np.save('test_loss.npy', test_loss)
        np.save('test_acc.npy', test_acc)
        # Print progress info.
        print ('alpha[%f]: train_loss=%f, train_acc=%f, test_loss=%f, test_acc=%f' % (alpha, train_loss[index], train_acc[index], test_loss[index], test_acc[index]))