# #!/bin/bash
# python3 SINC_PLOT.py 

import sys
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    

    # Save history of acc to npy file.
    train_loss_history_Sinc0 = np.load('train_loss_history_Sinc0.npy')
    train_loss_history_Sinc1 = np.load('train_loss_history_Sinc1.npy')
    train_loss_history_Sinc2 = np.load('train_loss_history_Sinc2.npy')

    # function define.
    x = np.linspace(0.0, 1.0, 100)
    y = np.sinc(5 * x)
    # Load model from h5 file.
    model0 = load_model('Sinc0.h5')
    model1 = load_model('Sinc1.h5')
    model2 = load_model('Sinc2.h5')
    # predict the results from model0~2.
    y_pred0 = np.squeeze(model0.predict(x))
    y_pred1 = np.squeeze(model1.predict(x))
    y_pred2 = np.squeeze(model2.predict(x))

    # # # Remove plt before summit to github.
    
    import matplotlib.lines as mlines
    # # Force matplotlib to not use any Xwindows backend.
    # plt.switch_backend('agg')
    # Summarize history for loss.
    plt.plot(train_loss_history_Sinc0)
    plt.plot(train_loss_history_Sinc1)
    plt.plot(train_loss_history_Sinc2)
    plt.title('Loss v.s. Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('# of epoch')
    plt.legend(['model0_loss', 'model1_loss', 'model2_loss'],loc='lower left')
    plt.savefig('SINC_LOSS.png')

    plt.clf()

    plt.plot(x, y, color='black')
    plt.plot(x, y_pred0, color='red')
    plt.plot(x, y_pred1, color='green')
    plt.plot(x, y_pred2, color='blue')
    plt.title('Prediction')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(['sinc(5x)', 'model0', 'model1', 'model2'],
     loc='upper right')
    plt.savefig('SINC_Predict.png')

