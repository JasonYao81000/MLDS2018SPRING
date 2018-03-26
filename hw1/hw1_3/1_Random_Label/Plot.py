import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    train_acc_MNIST = np.load('train_acc_MNIST.npy')
    train_loss_MNIST = np.load('train_loss_MNIST.npy')
    valid_acc_MNIST = np.load('valid_acc_MNIST.npy')
    valid_loss_MNIST = np.load('valid_loss_MNIST.npy')

    # i = np.argmax(loss)
    # maxLoss = np.max(loss)
    # print(i, maxLoss)
    # exit()

    plt.plot(train_acc_MNIST)
    plt.plot(valid_acc_MNIST)
    plt.title('Accuracy v.s. Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['train_acc', 'test_acc'],
     loc='lower right')
    plt.savefig('MNIST_ACC.png')

    plt.clf()

    plt.plot(train_loss_MNIST)
    plt.plot(valid_loss_MNIST)
    plt.title('Loss v.s. Epoch')
    plt.ylabel('Loss')
    plt.xlabel('# of epoch')
    plt.legend(['train_loss', 'test_loss'],
     loc='lower right')
    plt.savefig('MNIST_LOSS.png')