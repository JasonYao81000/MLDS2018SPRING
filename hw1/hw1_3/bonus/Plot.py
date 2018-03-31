import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":
    # batch size of each model.
    batch_size = np.load('batch_size.npy')

    # Load sharpness, loss and Acc. from npy file.
    sharpness = np.load('sharpness.npy')
    train_loss = np.load('train_loss.npy')
    train_acc = np.load('train_acc.npy')
    test_loss = np.load('test_loss.npy')
    test_acc = np.load('test_acc.npy')

    plt.figure()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('batchsize (log scale)')
    ax1.set_ylabel('Loss', color='b')
    ax1.plot(batch_size, train_loss, 'b')
    ax1.plot(batch_size, test_loss, 'b--')
    ax1.set_xscale("log", nonposx='clip')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(['train_loss', 'test_loss'], loc='upper left')

    # Instantiate a second axes that shares the same x-axis.
    ax2 = ax1.twinx()

    ax2.set_ylabel('Sharpness', color='r')
    ax2.plot(batch_size, sharpness, 'r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(['sharpness'], loc='upper right')

    plt.title('loss, sharpness v.s. batch_size')
    # Otherwise the right y-label is slightly clipped.
    fig.tight_layout()
    fig.savefig('Loss.png')

    plt.figure()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('batchsize (log scale)')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.plot(batch_size, train_acc, 'b')
    ax1.plot(batch_size, test_acc, 'b--')
    ax1.set_xscale("log", nonposx='clip')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(['train_acc', 'test_acc'], loc='upper left')

    # Instantiate a second axes that shares the same x-axis.
    ax2 = ax1.twinx()

    ax2.set_ylabel('Sharpness', color='r')
    ax2.plot(batch_size, sharpness, 'r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(['sharpness'], loc='upper right')

    plt.title('accuracy, sharpness v.s. batch_size')
    # Otherwise the right y-label is slightly clipped.
    fig.tight_layout()
    fig.savefig('Accuracy.png')