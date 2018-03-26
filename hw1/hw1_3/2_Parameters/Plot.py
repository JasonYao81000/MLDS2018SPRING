import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":

    # Load parameter, loss and Acc. from npy file.
    model_parameters = np.load('model_parameters.npy')
    train_loss = np.load('train_loss.npy')
    train_acc = np.load('train_acc.npy')
    test_loss = np.load('test_loss.npy')
    test_acc = np.load('test_acc.npy')

    plt.clf()
    plt.scatter(model_parameters, train_loss)
    plt.scatter(model_parameters, test_loss)
    plt.title('Loss v.s. Parameter')
    plt.ylabel('Loss')
    plt.xlabel('# of parameters')
    plt.legend(['train_loss', 'test_loss'],
     loc='lower right')
    plt.savefig('MNIST_LOSS.png')

    plt.clf()
    plt.scatter(model_parameters, train_acc)
    plt.scatter(model_parameters, test_acc)
    plt.title('Accuracy v.s. Parameter')
    plt.ylabel('Accuracy')
    plt.xlabel('# of parameters')
    plt.legend(['train_acc', 'test_acc'],
     loc='lower right')
    plt.savefig('MNIST_ACC.png')
