# #!/bin/bash
# python3 CNN_MNIST_PLOT.py 

import sys
import numpy as np

if __name__ == "__main__":
    

    # Save history of acc to npy file.
    train_acc_history_CNN_mnist_model0 = np.load('train_acc_history_CNN_mnist_model0.npy')
    train_loss_history_CNN_mnist_model0 = np.load('train_loss_history_CNN_mnist_model0.npy')

    train_acc_history_CNN_mnist_model1 = np.load('train_acc_history_CNN_mnist_model1.npy')
    train_loss_history_CNN_mnist_model1 = np.load('train_loss_history_CNN_mnist_model1.npy')

    train_acc_history_CNN_mnist_model2 = np.load('train_acc_history_CNN_mnist_model2.npy')
    train_loss_history_CNN_mnist_model2 = np.load('train_loss_history_CNN_mnist_model2.npy')

    

    # # # Remove plt before summit to github.
    import matplotlib.pyplot as plt
    # # Force matplotlib to not use any Xwindows backend.
    # plt.switch_backend('agg')
    # Summarize history for accuracy
    plt.plot(train_acc_history_CNN_mnist_model0)
    plt.plot(train_acc_history_CNN_mnist_model1)
    plt.plot(train_acc_history_CNN_mnist_model2)
    plt.title('Accuracy v.s. Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['CNN_mnist_model0', 'CNN_mnist_model1', 'CNN_mnist_model2'],
     loc='lower right')
    plt.savefig('CNN_MNIST_ACC.png')

    plt.clf()

    plt.plot(train_loss_history_CNN_mnist_model0)
    plt.plot(train_loss_history_CNN_mnist_model1)
    plt.plot(train_loss_history_CNN_mnist_model2)
    plt.title('Loss v.s. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('# of epoch')
    plt.legend(['CNN_mnist_model0', 'CNN_mnist_model1', 'CNN_mnist_model2'],
     loc='upper right')
    plt.savefig('CNN_MNIST_LOSS.png')

