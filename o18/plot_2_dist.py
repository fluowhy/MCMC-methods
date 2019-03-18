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

directory1 = "/home/mauricio/Downloads/final_results/l2hmc_10k"
method1 = "l2hmc"

directory2 = "/home/mauricio/Downloads/final_results/hmc_10k"
method2 = "hmc"

savedirec = "/home/mauricio/Downloads/final_results/mix_dist_l2hmc_hmc"

#chains = np.load("{}/chain_points_{}.npy".format(directory, method))
chains1 = np.load("{}/chain_points.npy".format(directory1))
chains2 = np.load("{}/chain_points.npy".format(directory2))
#hp = np.genfromtxt("{}/hyperparameters.txt".format(directory))
batch_size = 40# int(hp[2, 1])

# remove nan
chains1 = RemoveNan(chains1)
chains2 = RemoveNan(chains2)

# remove zero var
chains1 = RemoveZeroVar(chains1)
chains2 = RemoveZeroVar(chains2)

# remove burn in
chains1 = BurnInOut(chains1, 100)
chains2 = BurnInOut(chains2, 100)

labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'w']

for i in range(batch_size):
	try:
		plt.clf()
		t11 = chains1[:, i, 0]
		t21 = chains1[:, i, 1]
		t31 = chains1[:, i, 2]
		samps1 = np.vstack((t11, t21, t31)).T
		t12 = chains2[:, i, 0]
		t22 = chains2[:, i, 1]
		t32 = chains2[:, i, 2]
		samps2 = np.vstack((t12, t22, t32)).T
		samples1 = MCSamples(samples=samps1, names=labs1, labels=labs1, label=method1)
		samples2 = MCSamples(samples=samps2, names=labs1, labels=labs1, label=method2)
		#Triangle plot
		g = plots.getSubplotPlotter()
		samples1.updateSettings({'contours': [0.95, 0.99]})
		samples2.updateSettings({'contours': [0.95, 0.99]})
		g.settings.num_plot_contours = 2
		g.triangle_plot([samples1, samples2], filled=True)
		plt.savefig("{}/bi_dist_chain_{}".format(savedirec, i), dpi=dpi)
	except:
		print("Cadena {} tuvo error.".format(i))