import numpy as np
import matplotlib.pyplot as plt
from gelman_rubin import *
from autocorrelation import *


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

# upload chains
chains_l2hmc = np.load("/home/mauricio/Downloads/final_results/l2hmc_10k/chain_points.npy")
chains_hmc = np.load("/home/mauricio/Downloads/final_results/hmc_10k/chain_points.npy")
chains_mh = np.load("/home/mauricio/Downloads/final_results/mh_10k/chain_points.npy")

savedirec = "/home/mauricio/Downloads/final_results/l2hmc_10k/results/"

# remove nan
chains_l2hmc = RemoveNan(chains_l2hmc)
chains_hmc = RemoveNan(chains_hmc)
chains_mh = RemoveNan(chains_mh)

# remove zero var
chains_l2hmc = RemoveZeroVar(chains_l2hmc)
chains_hmc = RemoveZeroVar(chains_hmc)
chains_mh = RemoveZeroVar(chains_mh)

# remove burn in
chains_l2hmc = BurnInOut(chains_l2hmc, 100)
chains_hmc = BurnInOut(chains_hmc, 100)
chains_mh = BurnInOut(chains_mh, 100)


auto_l2hmc = autocorrelationvect(chains_l2hmc, kmax=100)
auto_hmc = autocorrelationvect(chains_hmc, kmax=100)
auto_mh = autocorrelationvect(chains_mh, kmax=100)

auto_l2hmc = np.mean(auto_l2hmc, axis=1)
auto_hmc = np.mean(auto_hmc, axis=1)
auto_mh = np.mean(auto_mh, axis=1)

names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]

plt.clf()
plt.plot(auto_l2hmc[:, 0], label="L2HMC")
plt.plot(auto_hmc[:, 0], label="HMC")
plt.plot(auto_mh[:, 0], label="MH")
plt.legend()
plt.xlabel("lag-k")
plt.ylabel("autocorrelacion")
plt.title(r'$\Omega_{m}$')
plt.savefig("{}auto1".format(savedirec), dpi=dpi)

plt.clf()
plt.plot(auto_l2hmc[:, 1], label="L2HMC")
plt.plot(auto_hmc[:, 1], label="HMC")
plt.plot(auto_mh[:, 1], label="MH")
plt.legend()
plt.xlabel("lag-k")
plt.ylabel("autocorrelacion")
plt.title(r'$\Omega_{\Lambda}$')
plt.savefig("{}auto2".format(savedirec), dpi=dpi)

plt.clf()
plt.plot(auto_l2hmc[:, 2], label="L2HMC")
plt.plot(auto_hmc[:, 2], label="HMC")
plt.plot(auto_mh[:, 2], label="MH")
plt.legend()
plt.xlabel("lag-k")
plt.ylabel("autocorrelacion")
plt.title(r'$w$')
plt.savefig("{}auto3".format(savedirec), dpi=dpi)


#################################################3

n, m, dim = chains_l2hmc.shape
tmh = theta_m_hat(chains_l2hmc)
smh = sigma_m_hat(chains_l2hmc)
th = theta_hat(tmh)
B = b(tmh, th, n, m)
W = w(smh)
vh = v_hat(W, B, n, m)
dc = dcorr(vh)
rc_l2hmc = psrf(vh, W)
rc_l2hmc_d = psrf(vh, W, dc, True)

n, m, dim = chains_hmc.shape
tmh = theta_m_hat(chains_hmc)
smh = sigma_m_hat(chains_hmc)
th = theta_hat(tmh)
B = b(tmh, th, n, m)
W = w(smh)
vh = v_hat(W, B, n, m)
dc = dcorr(vh)
rc_hmc = psrf(vh, W)
rc_hmc_d = psrf(vh, W, dc, True)

n, m, dim = chains_mh.shape
tmh = theta_m_hat(chains_mh)
smh = sigma_m_hat(chains_mh)
th = theta_hat(tmh)
B = b(tmh, th, n, m)
W = w(smh)
vh = v_hat(W, B, n, m)
dc = dcorr(vh)
rc_mh = psrf(vh, W)
rc_mh_d = psrf(vh, W, dc, True)

R = np.vstack((rc_mh, rc_hmc, rc_l2hmc))
Rd = np.vstack((rc_mh_d, rc_hmc_d, rc_l2hmc_d))
plt.clf()
PlotGR(R, "{}gelmanrubin".format(savedirec))
plt.clf()
PlotGR(Rd, "{}gelmanrubind".format(savedirec))