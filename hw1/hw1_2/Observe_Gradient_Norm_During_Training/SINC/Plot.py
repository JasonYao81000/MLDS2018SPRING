import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    gradient = np.load('gradientNorm.npy')
    loss = np.load('loss.npy')

    plt.subplot(211)
    plt.plot(gradient)
    # plt.title('gradientNorm')
    plt.ylabel('grad')


    plt.subplot(212)
    plt.plot(loss)
    # plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')

    plt.savefig('gradientNorm.png')