# #!/bin/bash
# python3 CNN_CIFAR10_PLOT.py 

import sys
import numpy as np

if __name__ == "__main__":
    

    # Save history of acc to npy file.
    train_acc_history_CNN_model0 = np.load('train_acc_history_CNN_model0.npy')
    train_loss_history_CNN_model0 = np.load('train_loss_history_CNN_model0.npy')
    test_acc_history_CNN_model0 = np.load('test_acc_history_CNN_model0.npy')
    test_loss_history_CNN_model0 = np.load('test_loss_history_CNN_model0.npy')

    train_acc_history_CNN_model1 = np.load('train_acc_history_CNN_model1.npy')
    train_loss_history_CNN_model1 = np.load('train_loss_history_CNN_model1.npy')
    test_acc_history_CNN_model1 = np.load('test_acc_history_CNN_model1.npy')
    test_loss_history_CNN_model1 = np.load('test_loss_history_CNN_model1.npy')

    train_acc_history_CNN_model2 = np.load('train_acc_history_CNN_model2.npy')
    train_loss_history_CNN_model2 = np.load('train_loss_history_CNN_model2.npy')
    test_acc_history_CNN_model2 = np.load('test_acc_history_CNN_model2.npy')
    test_loss_history_CNN_model2 = np.load('test_loss_history_CNN_model2.npy')

    # train_acc_history_CNN_model3 = np.load('train_acc_history_CNN_model3.npy')
    # train_loss_history_CNN_model3 = np.load('train_loss_history_CNN_model3.npy')
    # test_acc_history_CNN_model3 = np.load('test_acc_history_CNN_model3.npy')
    # test_loss_history_CNN_model3 = np.load('test_loss_history_CNN_model3.npy')

    # # # Remove plt before summit to github.
    import matplotlib.pyplot as plt
    # # Force matplotlib to not use any Xwindows backend.
    # plt.switch_backend('agg')
    # Summarize history for accuracy
    plt.plot(train_acc_history_CNN_model0)
    plt.plot(test_acc_history_CNN_model0)
    plt.plot(train_acc_history_CNN_model1)
    plt.plot(test_acc_history_CNN_model1)
    plt.plot(train_acc_history_CNN_model2)
    plt.plot(test_acc_history_CNN_model2)
    # plt.plot(train_acc_history_CNN_model3)
    # plt.plot(test_acc_history_CNN_model3)
    plt.title('Accuracy v.s. Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['model0_train', 'model0_test', 'model1_train', 'model1_test', 'model2_train', 'model2_test'],
     loc='lower right')
    plt.savefig('CNN_CIFAR10_ACC.png')

    plt.clf()

    plt.plot(train_loss_history_CNN_model0)
    plt.plot(test_loss_history_CNN_model0)
    plt.plot(train_loss_history_CNN_model1)
    plt.plot(test_loss_history_CNN_model1)
    plt.plot(train_loss_history_CNN_model2)
    plt.plot(test_loss_history_CNN_model2)
    # plt.plot(train_loss_history_CNN_model3)
    # plt.plot(test_loss_history_CNN_model3)
    plt.title('Loss v.s. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('# of epoch')
    plt.legend(['model0_train', 'model0_test', 'model1_train', 'model1_test', 'model2_train', 'model2_test'],
     loc='upper right')
    plt.savefig('CNN_CIFAR10_LOSS.png')

