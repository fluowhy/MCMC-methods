import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples


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


def Concat(chain):
	"""Concatenate chains.
	chain: numpy array, chain.
	"""
	a, b, c = chain.shape
	return np.reshape(chain, (a*b, c))


global dpi
dpi = 200

directory1 = "/home/mauricio/Downloads/final_results/l2hmc_10k"
method1 = "l2hmc"
directory2 = "/home/mauricio/Downloads/final_results/mh_10k"
method2 = "mh"
savedirec = "/home/mauricio/Downloads/final_results/dist_concat"

#chains = np.load("{}/chain_points_{}.npy".format(directory, method))
chains1 = np.load("{}/chain_points.npy".format(directory1))
chains2 = np.load("{}/chain_points.npy".format(directory2))
#hp = np.genfromtxt("{}/hyperparameters.txt".format(directory))

# remove nan
chains1 = RemoveNan(chains1)
chains2 = RemoveNan(chains2)

# remove zero var
chains1 = RemoveZeroVar(chains1)
chains2 = RemoveZeroVar(chains2)

# remove burn in
chains1 = BurnInOut(chains1, 200)
chains2 = BurnInOut(chains2, 200)

# concatenate
newchain1 = Concat(chains1)
newchain2 = Concat(chains2)

labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'w']

plt.clf()
samples1 = MCSamples(samples=newchain1, names=labs1, labels=labs1, label=method1)
samples2 = MCSamples(samples=newchain2, names=labs1, labels=labs1, label=method2)
g = plots.getSubplotPlotter()
samples1.updateSettings({'contours': [0.68, 0.95, 0.99], 'fine_bins_2D': 64})
samples2.updateSettings({'contours': [0.68, 0.95, 0.99], 'fine_bins_2D': 64})
g.settings.num_plot_contours = 3
g.triangle_plot([samples1, samples2], filled=True)
plt.savefig("{}/{}_{}".format(savedirec, method1, method2), dpi=dpi)
