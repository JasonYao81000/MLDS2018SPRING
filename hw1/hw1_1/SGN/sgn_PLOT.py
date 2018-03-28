# #!/bin/bash
# python3 sgn_PLOT.py 

import sys
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    

    # Save history of acc to npy file.
    train_loss_history_Sgn0 = np.load('train_loss_history_Sgn0.npy')
    train_loss_history_Sgn1 = np.load('train_loss_history_Sgn1.npy')
    train_loss_history_Sgn2 = np.load('train_loss_history_Sgn2.npy')

    # function define.
    x = np.linspace(0.01, 1.0, 100)
    y = np.sign(np.sin(5 * np.pi * x))
    # Load model from h5 file.
    model0 = load_model('Sgn0.h5')
    model1 = load_model('Sgn1.h5')
    model2 = load_model('Sgn2.h5')
    # predict the results from model0~2.
    y_pred0 = np.squeeze(model0.predict(x))
    y_pred1 = np.squeeze(model1.predict(x))
    y_pred2 = np.squeeze(model2.predict(x))

    # # # Remove plt before summit to github.
    
    import matplotlib.lines as mlines
    # # Force matplotlib to not use any Xwindows backend.
    # plt.switch_backend('agg')
    # Summarize history for loss.
    plt.plot(train_loss_history_Sgn0)
    plt.plot(train_loss_history_Sgn1)
    plt.plot(train_loss_history_Sgn2)
    plt.title('Loss v.s. Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('# of epoch')
    plt.legend(['model0_loss', 'model1_loss', 'model2_loss'],loc='lower left')
    plt.savefig('SGN_LOSS.png')

    plt.clf()

    plt.plot(x, y, color='black')
    plt.plot(x, y_pred0, color='red')
    plt.plot(x, y_pred1, color='green')
    plt.plot(x, y_pred2, color='blue')
    plt.title('Prediction')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(['sgn(sin(5πx))', 'model0', 'model1', 'model2'],
     loc='upper right')
    plt.savefig('SGN_Predict.png')

