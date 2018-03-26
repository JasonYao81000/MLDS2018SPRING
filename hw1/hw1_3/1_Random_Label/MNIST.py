# python3 MNIST.py 

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
    
    batch_size = 128
    num_classes = 10
    epochs = 94879487
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

    # Random shuffle y_data_train.
    np.random.shuffle(y_data_train)

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
    model.add(Conv2D(16, (3, 3), kernel_initializer='lecun_normal', padding='SAME', bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Conv2D(32, (3, 3), kernel_initializer='lecun_normal', padding='SAME', bias_initializer='zeros'))
    model.add(Activation('selu'))


    # Fully-connected classifier.
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='lecun_normal', bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer='lecun_normal',bias_initializer='zeros'))
    print ('Created the model.')
    print (model.summary())
    # # of parameteres: 128,150
    # exit()
    # Compile the model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Compiled the model.')

    earlyStopping = EarlyStopping(monitor='acc', patience=15, verbose=1, mode='auto')

    # Fit the model.
    fitHistory = model.fit(x_data_train, y_data_train,
        batch_size=batch_size, 
        epochs=epochs,
        # shuffle = True,
        validation_data=(x_data_test, y_data_test), 
        callbacks=[earlyStopping])

    # Save model to h5 file.
    model.save('MNIST.h5')

    # Save history of acc to npy file.
    np.save('train_acc_MNIST.npy', fitHistory.history['acc'])
    np.save('train_loss_MNIST.npy', fitHistory.history['loss'])
    np.save('valid_acc_MNIST.npy', fitHistory.history['val_acc'])
    np.save('valid_loss_MNIST.npy', fitHistory.history['val_loss'])

    # # Report index of highest accuracy in training set and validation set.
    # print ('tra_acc: ', np.amax(fitHistory.history['acc']), 'at epochs = ', np.argmax(fitHistory.history['acc']))

    # # # Remove plt before summit to github.
    # import matplotlib.pyplot as plt
    # # # Force matplotlib to not use any Xwindows backend.
    # # plt.switch_backend('agg')
    # # Summarize history for accuracy
    # plt.plot(fitHistory.history['acc'])
    # # plt.plot(fitHistory.history['val_acc'])
    # plt.title('Accuracy v.s. Epoch')
    # plt.ylabel('Accuracy')
    # plt.xlabel('# of epoch')
    # # plt.legend(['train', 'valid'], loc='lower right')
    # plt.savefig('Accuracy v.s. Epoch.png')