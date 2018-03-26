import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":

    # Interpolation ratio.
    alphas = np.linspace(start=-1, stop=2, num=50)

    # Load loss and Acc. from npy file.
    train_loss = np.load('train_loss.npy')
    train_acc = np.load('train_acc.npy')
    test_loss = np.load('test_loss.npy')
    test_acc = np.load('test_acc.npy')

    plt.clf()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Loss', color='b')
    ax1.plot(alphas, train_loss, 'b')
    ax1.plot(alphas, test_loss, 'b--')
    ax1.set_yscale("log", nonposx='clip')
    ax1.tick_params(axis='y', labelcolor='b')

    # Instantiate a second axes that shares the same x-axis.
    ax2 = ax1.twinx()

    ax2.set_ylabel('Accuracy', color='r')
    ax2.plot(alphas, train_acc, 'r')
    ax2.plot(alphas, test_acc, 'r--')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(['train', 'test'], loc='upper right')

    # Otherwise the right y-label is slightly clipped.
    fig.tight_layout()
    fig.savefig('BatchSize.png')