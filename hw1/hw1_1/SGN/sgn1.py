# #!/bin/bash
# python3 sgn1.py 

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

    X_train = np.linspace(0.01, 1.0, num=10000)
    y_train = np.sign(np.sin(5 * np.pi * X_train))
    
    model = Sequential()
    model.add(Dense(10, input_dim=1, kernel_initializer='normal', activation='selu'))
    model.add(Dense(18, kernel_initializer='normal', activation='selu'))
    model.add(Dense(15, kernel_initializer='normal', activation='selu'))
    model.add(Dense(4, kernel_initializer='normal', activation='selu'))
    model.add(Dense(1,kernel_initializer='normal'))
    print ('Created the model.')
    print (model.summary())

    model.compile(loss='mse', optimizer='adam')

    fitHistory = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle = True)
    # score = model.evaluate(X_test, y_test, batch_size=batch_size)

    # Save model to h5 file.
    model.save('Sgn1.h5')

    # Save history of acc to npy file.
    np.save('train_loss_history_Sgn1.npy', fitHistory.history['loss'])

    # import matplotlib.pyplot as plt
    # # # # Force matplotlib to not use any Xwindows backend.
    # # # plt.switch_backend('agg')
    # # # Summarize history for accuracy
    # plt.plot(X_train, y_train)
    # plt.show()