# python3 MNIST_1024.py 

import sys
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras import backend as K
from keras.datasets import mnist

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')
    
    batch_size = 1024
    num_classes = 10
    epochs = 20
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

    # Create the model.
    model = Sequential()
    # Conv block 1: 64 output filters.
    model.add(Conv2D(16, kernel_size=(3, 3),
                activation='selu',
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                padding='SAME',
                input_shape=input_shape))

    # Fully-connected classifier.
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_initializer='lecun_normal',bias_initializer='zeros'))
    print ('Created the model.')
    print (model.summary())
    print (model.count_params())

    # Compile the model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Compiled the model.')

    # Fit the model.
    fitHistory = model.fit(x_data_train, y_data_train,
        batch_size=batch_size, 
        epochs=epochs,
        # shuffle = True,
        validation_data=(x_data_test, y_data_test))

    # Save model to h5 file.
    model.save('MNIST_1024.h5')