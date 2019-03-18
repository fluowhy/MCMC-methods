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


global dpi
dpi = 200

directory = "/home/mauricio/Documents/Uni/primavera_2018/cosmo/results1/"
method = "l2hmc"

#chains = np.load("{}/chain_points_{}.npy".format(directory, method))
chains = np.load("{}chain_points.npy".format(directory))
#hp = np.genfromtxt("{}/hyperparameters.txt".format(directory))
batch_size = 40# int(hp[2, 1])

# remove nan
chains = RemoveNan(chains)

# remove zero var
chains = RemoveZeroVar(chains)

# remove burn in
chains = BurnInOut(chains, 100)

labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'w']

for i in range(batch_size):
	try:
		plt.clf()
		smooth = 0.5
		t1 = chains[:, i, 0]
		t2 = chains[:, i, 1]
		t3 = chains[:, i, 2]
		samps = np.vstack((t1, t2, t3)).T
		samples = MCSamples(samples=samps, names=labs1, labels=labs1)
		#Triangle plot
		g = plots.getSubplotPlotter()
		samples.updateSettings({'contours': [0.68, 0.95, 0.99], "smooth_scale_1D": smooth, "smooth_scale_2D": smooth})
		g.settings.num_plot_contours = 2
		g.triangle_plot([samples], filled=True)
		plt.savefig("{}dist_chain_{}".format(directory, i), dpi=dpi)
	except:
		print("Cadena {} tuvo error.".format(i))