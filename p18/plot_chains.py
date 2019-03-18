import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

global dpi
dpi = 200


def BurnInOut(chain, fromsample):
    """
    Remove burn in.
    chain: numpy array, chain.
    fromsample: int, sample number to remove.
    """
    return chain[fromsample:, :, :]


def RemoveNan(chain):
    """
    Remove nan.
    chain: numpy array, chain.
    """
    return chain[:, np.invert(np.isnan(np.sum(np.sum(chain, axis=0), axis=1))), :]


def RemoveZeroVar(chain):
    """
    Remove chain if zero var.
    chain: numpy array, chain.
    """
    return chain[:, np.invert((np.sum(np.var(chain, axis=0), axis=1)<1e-10)), :]


def Plot(arr1, arr2, arr3, names, save=False, savename=None, savename_title=None, alpha=0.5):
    plt.clf()
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 12))
    ax1 = ax[0, 0]
    ax2 = ax[1, 0]
    ax3 = ax[1, 1]
    ax1.scatter(arr1, arr2, color='navy', marker='.', alpha=alpha)
    #ax1.scatter(np.mean(arr1), np.mean(arr2), color='black', marker='+', label='expected value')
    #ax1.scatter(arr1[0], arr2[0], color='black', label='initial state')
    ax2.scatter(arr1, arr3, color='red', marker='.', alpha=alpha)
    #ax2.scatter(np.mean(arr1), np.mean(arr3), color='black', marker='+')
    #ax2.scatter(arr1[0], arr3[0], color='black')
    ax3.scatter(arr2, arr3, color='green', marker='.', alpha=alpha)
    #ax3.scatter(np.mean(arr2), np.mean(arr3), color='black', marker='+')
    #ax3.scatter(arr2[0], arr3[0], color='black')
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


if __name__ == "__main__":
    directory = "results9/"
    name = "samples.npy"
    method = "l2hmc"

    # pre processing
    Points = np.load("{}{}".format(directory, name))
    loss = np.load("{}loss_train.npy".format(directory))

    # remove nan
    #Points = RemoveNan(Points)

    # remove zero var
    #Points = RemoveZeroVar(Points)

    # remove burn in
    #Points = BurnInOut(Points, 100)

    _, batch_size, _ = Points.shape
    smooth = -1

    # samples plot
    names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]
    labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'\omega']
    maxs = np.max(np.max(Points, axis=0), axis=0)
    mins = np.min(np.min(Points, axis=0), axis=0)
    for i in range(batch_size):
        print("Chain {} de {}".format(i, batch_size))
        #det = np.linalg.det(np.cov(Points[:, i, :].T))
        #if det>1e-7:
        #print(i, "det =", det)
        Plot(Points[:, i, 0], Points[:, i, 1], Points[:, i, 2], names, save=True, savename='{}samples_{}.png'.format(directory, i), savename_title=method)


        # dist plot

        """
        plt.clf()
        samples = MCSamples(samples=Points[:, i, :], names=labs1, labels=labs1)
        g = plots.getSubplotPlotter()
        samples.updateSettings({'contours': [0.68, 0.95, 0.99], 'fine_bins_2D': 256, "smooth_scale_1D": smooth, "smooth_scale_2D": smooth})
        g.settings.num_plot_contours = 2
        g.triangle_plot([samples], filled=True, param_limits={labs1[0]: [0, 1], labs1[1]: [0, 1], labs1[2]: [-4, -1/3]})
        plt.savefig("{}dist_{}".format(directory, i), dpi=dpi)
        """

    #plt.clf()
    #plt.plot(loss, color="black")
    #plt.xlabel("chain step")
    #plt.ylabel("value")
    #plt.title("training loss")
    #plt.savefig("{}loss".format(directory), dpi=dpi)