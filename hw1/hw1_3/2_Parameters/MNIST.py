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
    
    batch_size = 1024
    num_classes = 10
    epochs = 20
    num_models = 50
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

    # # Random shuffle y_data_train.
    # np.random.shuffle(y_data_train)

    # Convert labels to one hot encoding.
    y_data_train = to_categorical(y_data_train)
    y_data_test = to_categorical(y_data_test)

    # Parameters of each model.
    model_parameters = np.zeros(num_models)
    # Loss and Acc. of each model in training set.
    train_loss = np.zeros(num_models)
    train_acc = np.zeros(num_models)
    # Loss and Acc. of each model in testimg set.
    test_loss = np.zeros(num_models)
    test_acc = np.zeros(num_models)

    for i in range(num_models):
        # Create the model.
        model = Sequential()
        # Conv block 1: 64 output filters.
        model.add(Conv2D(i + 1, kernel_size=(3, 3),
                    activation='selu',
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    padding='SAME',
                    input_shape=input_shape))

        # Fully-connected classifier.
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',kernel_initializer='lecun_normal',bias_initializer='zeros'))
        # print ('Created the model.')
        # print (model.summary())
        # print (model.count_params())

        # Compile the model.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print ('Compiled the model.')

        # Fit the model.
        fitHistory = model.fit(x_data_train, y_data_train,
            batch_size=batch_size, 
            epochs=epochs,
            # shuffle = True,
            validation_data=(x_data_test, y_data_test))

        # Save model to h5 file.
        model.save('MNIST_%03d.h5' % i)
        # Record parameters of each model.
        model_parameters[i] = model.count_params()
        # Record loss and Acc. of each model in training set.
        train_loss[i] = fitHistory.history['loss'][-1]
        train_acc[i] = fitHistory.history['acc'][-1]
        # Record loss and Acc. of each model in testimg set.
        test_loss[i] = fitHistory.history['val_loss'][-1]
        test_acc[i] = fitHistory.history['val_acc'][-1]
        print ('Saved model %03d with %d paramters.' % (i, model_parameters[i]))
        print ('Train Loss: %f, Acc: %f' %(train_loss[i], train_acc[i]))
        print ('Test Loss: %f, Acc: %f' %(test_loss[i], test_acc[i]))

        # Save parameter, loss and Acc. to npy file.
        np.save('model_parameters.npy', model_parameters)
        np.save('train_loss.npy', train_loss)
        np.save('train_acc.npy', train_acc)
        np.save('test_loss.npy', test_loss)
        np.save('test_acc.npy', test_acc)

    # # Save history of acc to npy file.
    # np.save('train_acc_MNIST.npy', fitHistory.history['acc'])
    # np.save('train_loss_MNIST.npy', fitHistory.history['loss'])
    # np.save('valid_acc_MNIST.npy', fitHistory.history['val_acc'])
    # np.save('valid_loss_MNIST.npy', fitHistory.history['val_loss'])

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