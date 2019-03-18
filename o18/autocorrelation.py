import numpy as np
import matplotlib.pyplot as plt

global dpi
dpi = 200


def plot(arr1, arr2, arr3, names, save=False, savename=None):
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
    ax1.set_title(savename)
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


def plot_autocorr(c, names, savename,):
	numc = c.shape[1]
	plt.clf()
	for i in range(numc):
		if i==0:
			plt.plot(c[:, i, 0], color='navy', alpha=0.5, label=names[0])
			plt.plot(c[:, i, 1], color='orange', alpha=0.5, label=names[1])
			plt.plot(c[:, i, 2], color='green', alpha=0.5, label=names[2])
		else:
			plt.plot(c[:, i, 0], color='navy', alpha=0.5)
			plt.plot(c[:, i, 1], color='orange', alpha=0.5)
			plt.plot(c[:, i, 2], color='green', alpha=0.5)
	plt.xlabel('k-lag')
	plt.ylabel('autocorrelation')
	plt.legend()
	plt.title(savename)
	plt.savefig('{}_autocorr'.format(savename), dpi=dpi)

	return


def plot_autocorr_mean(c, names, savename):
	plt.clf()
	cmean = np.mean(c, axis=1)
	cstd = np.std(c, axis=1)

	plt.plot(cmean[:, 0], color='navy', alpha=1, label=names[0])
	plt.plot(cmean[:, 1], color='orange', alpha=1, label=names[1])
	plt.plot(cmean[:, 2], color='green', alpha=1, label=names[2])


	plt.xlabel('k-lag')
	plt.ylabel('autocorrelation')
	plt.legend()
	plt.title(savename)
	plt.savefig('{}_autocorr_mean'.format(savename), dpi=dpi)

	return


def autocorrelationvect(chain, kmax=None):
    N, numc, nump = chain.shape
    chainmean = np.mean(chain, axis=0)
    den = np.sum((chain - chainmean)**2, axis=0)    
    if kmax==None:
        kmax = int(N/2)
    c = []
    for i in range(kmax):
        if i==0:
            corr = np.ones((numc, nump))
        else:
            corr = np.sum((chain[i:, :, :] - chainmean)*(chain[:-i] - chainmean), axis=0)/den
        c.append(corr)
    return np.array(c)

	
"""

#revisar cadenas validas

chains_l2hmc = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_l2hmc.npy")
chains_hmc = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_hmc.npy")
chains_mh = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_mh.npy")

# l2hmc chain 1
# hmc chain 9
# mh

# delete
# l2hmc 7, 8, 9, 10, 13, 19
# hmc 1, 4, 12, 16

# plot chains

ch_l2hmc = chains_l2hmc[:, 1, [0, 1]]
ch_hmc = chains_hmc[:, 9, [0, 1]]
ch_mh = chains_mh[:, 1, [0, 1]]


fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

plt.clf()
plt.scatter(ch_mh[:, 0], ch_mh[:, 1], color='navy', alpha=0.5, marker='.')
plt.title('MH')
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\Omega_{\Lambda}$")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("plot11", dpi=dpi)

plt.clf()
plt.scatter(ch_hmc[:, 0], ch_hmc[:, 1], color='navy', alpha=0.5, marker='.')
plt.title('HMC')
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\Omega_{\Lambda}$")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("plot12", dpi=dpi)

plt.clf()
plt.scatter(ch_l2hmc[:, 0], ch_l2hmc[:, 1], color='navy', alpha=0.5, marker='.')
plt.title('HMC*')
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\Omega_{\Lambda}$")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("plot13", dpi=dpi)


chains_l2hmc_auto = chains_l2hmc[:, [0, 1, 2, 3, 4, 5, 6, 11, 12, 14, 15, 16, 17, 18], :]
chains_hmc_auto = chains_hmc[:, [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
chains_mh_auto = chains_mh

auto_l2hmc = autocorrelationvect(chains_l2hmc_auto, kmax=500)
auto_hmc = autocorrelationvect(chains_hmc_auto, kmax=500)
auto_mh = autocorrelationvect(chains_mh_auto, kmax=500)

auto_l2hmc = np.mean(auto_l2hmc, axis=1)
auto_hmc = np.mean(auto_hmc, axis=1)
auto_mh = np.mean(auto_mh, axis=1)

names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]

plt.clf()
plt.plot(auto_l2hmc[:, 0], label="HMC*")
plt.plot(auto_hmc[:, 0], label="HMC")
plt.plot(auto_mh[:, 0], label="MH")
plt.legend()
plt.xlabel("k-lag")
plt.ylabel("autocorrelacion")
plt.title(r'$\Omega_{m}$')
plt.savefig("auto1", dpi=dpi)

plt.clf()
plt.plot(auto_l2hmc[:, 1], label="HMC*")
plt.plot(auto_hmc[:, 1], label="HMC")
plt.plot(auto_mh[:, 1], label="MH")
plt.legend()
plt.xlabel("k-lag")
plt.ylabel("autocorrelacion")
plt.title(r'$\Omega_{\Lambda}$')
plt.savefig("auto2", dpi=dpi)

plt.clf()
plt.plot(auto_l2hmc[:, 2], label="HMC*")
plt.plot(auto_hmc[:, 2], label="HMC")
plt.plot(auto_mh[:, 2], label="MH")
plt.legend()
plt.xlabel("k-lag")
plt.ylabel("autocorrelacion")
plt.title(r'$w$')
plt.savefig("auto3", dpi=dpi)


chain1 = np.load("/home/mauricio/Documents/Uni/Intro_2/mh_results/chain_points_mh.npy")
chain1 = [chain1[:, i, :] for i in range(chain1.shape[1]) ]
chain1 = np.array(chain1)
_, numc, _ = chain1.shape

C = autocorrelationvect(chain1)
names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]

plot_autocorr_mean(C, names, "mh")
plot_autocorr(C, names, "mh")

chain1 = np.load("/home/mauricio/Documents/Uni/Intro_2/mh_results/chain_points_hmc.npy")
_, numc, _ = chain1.shape

C = autocorrelationvect(chain1)
names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]

plot_autocorr_mean(C, names, "hmc")
plot_autocorr(C, names, "hmc")
"""