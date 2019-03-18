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

directory = "/home/mauricio/Downloads/final_results/mh_10k"
method = "mh"
savedirec = "/home/mauricio/Downloads/final_results/dist_concat"

#chains = np.load("{}/chain_points_{}.npy".format(directory, method))
chains = np.load("{}/chain_points.npy".format(directory))
#hp = np.genfromtxt("{}/hyperparameters.txt".format(directory))
batch_size = 40# int(hp[2, 1])

# remove nan
chains = RemoveNan(chains)

# remove zero var
chains = RemoveZeroVar(chains)

# remove burn in
chains = BurnInOut(chains, 200)

# concatenate
newchain = Concat(chains)

labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'w']

plt.clf()
samples = MCSamples(samples=newchain, names=labs1, labels=labs1)
g = plots.getSubplotPlotter()
samples.updateSettings({'contours': [0.68, 0.95, 0.99], 'fine_bins_2D': 64})
g.settings.num_plot_contours = 3
g.triangle_plot([samples], filled=True)
plt.savefig("{}/{}".format(savedirec, method), dpi=dpi)
