import numpy as np
import matplotlib.pyplot as plt

global dpi
dpi = 200

def Plot(arr1, arr2, arr3, names, save=False, savename=None, savename_title=None):
    plt.clf()
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 12))
    ax1 = ax[0, 0]
    ax2 = ax[1, 0]
    ax3 = ax[1, 1]
    ax1.scatter(arr1, arr2, color='navy', marker='.', alpha=0.1)
    ax1.scatter(np.mean(arr1), np.mean(arr2), color='black', marker='+', label='expected value')
    ax1.scatter(arr1[0], arr2[0], color='black', label='initial state')
    ax2.scatter(arr1, arr3, color='red', marker='.', alpha=0.1)
    ax2.scatter(np.mean(arr1), np.mean(arr3), color='black', marker='+')
    ax2.scatter(arr1[0], arr3[0], color='black')
    ax3.scatter(arr2, arr3, color='green', marker='.', alpha=0.1)
    ax3.scatter(np.mean(arr2), np.mean(arr3), color='black', marker='+')
    ax3.scatter(arr2[0], arr3[0], color='black')
    ax1.set_title(savename_title)
    # labels
    ax2.set_xlabel(names[0])
    ax1.set_ylabel(names[1])
    ax2.set_ylabel(names[2])
    ax3.set_xlabel(names[1])
    # limits
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-4, -1/3])
    ax3.set_xlim([0, 1])
    ax3.set_ylim([-4, -1/3])
    fig.legend()
    if save:
        fig.savefig(savename, dpi=dpi)
    return

directory = "/home/mauricio/Downloads/mh_10k"
method = "mh"

Points = np.load("{}/chain_points_{}.npy".format(directory, method))
#Points_train = np.load("{}/chain_points_train.npy".format(directory))
#hp = np.genfromtxt("{}/hyperparameters.txt".format(directory))
batch_size = 40#int(hp[2, 1])

names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]
maxs = np.max(np.max(Points, axis=0), axis=0)
mins = np.min(np.min(Points, axis=0), axis=0)
for i in range(batch_size):
    Plot(Points[:, i, 0], Points[:, i, 1], Points[:, i, 2], names, save=True, 
        savename='{}/samples/{}_{}.png'.format(directory, method, i), savename_title="l2hmc")

