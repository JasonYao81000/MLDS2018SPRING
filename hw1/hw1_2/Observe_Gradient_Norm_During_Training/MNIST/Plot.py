import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":

    gradient = np.load('gradientNorm.npy')
    loss = np.load('loss.npy')

    # i = np.argmax(loss)
    # maxLoss = np.max(loss)
    # print(i, maxLoss)
    # exit()

    plt.subplot(211)
    plt.plot(gradient)
    plt.text(2000, gradient[2000],
                '%.2f' % gradient[2000], 
                color='red')
    plt.text(8000, gradient[8000],
                '%.2f' % gradient[8000], 
                color='red')
    # plt.title('gradientNorm')
    plt.ylabel('grad')
    # plt.ylim([1.05, 1.2])

    plt.subplot(212)
    plt.plot(loss)
    # plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')

    plt.savefig('gradientNorm.png')