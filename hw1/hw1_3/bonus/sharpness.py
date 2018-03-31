# python3 sharpness.py 

import sys
import numpy as np
import random
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model
from keras.datasets import mnist

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    random.seed(9487)
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

    num_models = 20
    test_loss = np.load('test_loss.npy')
    sharpness = np.copy(test_loss)

    epsilon = 1e-4
    for model_num in range(num_models):
        # Load models from h5 file.
        model = load_model('MNIST_%03d.h5' %(model_num))
        # Get weights from model and convert to np.array.
        weights_init = np.array(model.get_weights())

        # number of the sample.
        sample_num = 1000
        for sample in range(sample_num):
            # Get weights from model and convert to np.array.
            weights = weights_init

            for index, w in enumerate(weights):
                offset = (random.random() - 0.5) * epsilon
                weights[index] += offset

            model.set_weights(list(weights))
            loss, acc = model.evaluate(x_data_test, y_data_test, batch_size=1024, verbose=0)
            if (loss > sharpness[model_num]):
                sharpness[model_num] = loss
            if (sample % (sample_num / 100) == 0):
                print ('Max. Loss[%06d]: %.6f' %(sample, sharpness[model_num]), end='\r')
        # Record sharpness.
        sharpness[model_num] -= test_loss[model_num]

        # Save sharpness to npy file.
        np.save('sharpness.npy', sharpness)
        # Print progress info.
        print ('\nModel[%03d]: sharpness=%f, test_loss=%f' % (model_num, sharpness[model_num], test_loss[model_num]))