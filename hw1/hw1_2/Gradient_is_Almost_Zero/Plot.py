import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":

    # Load loss and Acc. from npy file.
    loss = np.load('loss.npy')
    minimum_ratio = np.load('minimum_ratio.npy')

    plt.clf()
    plt.scatter(minimum_ratio, loss)
    plt.title('Loss v.s. Minimum ratio')
    plt.ylabel('Loss')
    plt.xlabel('minimum_ratio')
    plt.savefig('Loss_MinimumRatio.png')