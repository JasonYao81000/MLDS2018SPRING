# #!/bin/bash
# python3 sinc2.py 

import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    batch_size = 128
    epochs = 20000

    X_train = np.linspace(0.0001, 1.0, num=10000)
    y_train = np.sinc(5 * X_train)
    
    model = Sequential()
    model.add(Dense(190, input_dim=1, kernel_initializer='normal', activation='selu'))
    model.add(Dense(1,kernel_initializer='normal'))
    print ('Created the model.')
    print (model.summary())

    model.compile(loss='mse', optimizer='adam')

    fitHistory = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle = True)
    # score = model.evaluate(X_test, y_test, batch_size=batch_size)

    # Save model to h5 file.
    model.save('Sinc2.h5')

    # Save history of acc to npy file.
    np.save('train_loss_history_Sinc2.npy', fitHistory.history['loss'])

    # Report index of highest accuracy in training set and validation set.
    # print ('tra_acc: ', np.amax(fitHistory.history['acc']), 'at epochs = ', np.argmax(fitHistory.history['acc']))

    import matplotlib.pyplot as plt
    # # # Force matplotlib to not use any Xwindows backend.
    # # plt.switch_backend('agg')
    # # Summarize history for accuracy
    # plt.plot([X_train, y_train])
    # plt.show()