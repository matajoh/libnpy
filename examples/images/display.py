""" Python display script to test NPY/NPZ file output

This script assumes that you have the numpy and matplotlib modules installed
"""

import numpy as np
import matplotlib.pyplot as plt

def _main():
    plt.figure(1)

    # npy
    plt.subplot(131)
    plt.imshow(np.load("color.npy"))

    test = np.load("test.npz")

    # npz
    plt.subplot(132)
    plt.imshow(test["color.npy"])

    plt.subplot(133)
    plt.imshow(test["gray.npy"], cmap="gray")

    plt.show()


if __name__ == "__main__":
    _main()
