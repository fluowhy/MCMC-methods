import numpy as np
import matplotlib.pyplot as plt

global dpi
dpi = 200

def theta_m_hat(x):
    return np.mean(x, axis=0)   


def sigma_m_hat(x):
    return np.var(x, axis=0)


def theta_hat(x):
    return np.mean(x, axis=0)


def b(x1, x2, n, m):
    return n/(m - 1)*np.sum((x1 - x2)**2, axis=0)


def w(x):
    return np.mean(x, axis=0)


def v_hat(x1, x2, n, m):
    return (n - 1)*x1/n + (m + 1)/m/n*x2


def psrf(x1, x2, d=None, dcorr=False):
	r = np.sqrt(x1/x2)
	if dcorr:
		return np.sqrt((d + 3)/(d + 1))*r
	else:
		return r

def dcorr(x1):
	return 2*x1/np.var(x1)


def PlotGR(mat, savename):
    size = 3
    x_start = 0
    x_end = 3
    y_start = 0
    y_end = 3
    ep = 20
    jump_x = (x_end - x_start) / (ep * size)
    jump_y = (y_end - y_start) / (ep * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)    
    for y_index, yy in enumerate(y_positions):
        for x_index, xx in enumerate(x_positions):
            label = mat[y_index, x_index]
            text_x = xx + jump_x
            text_y = yy + jump_y
            plt.text(text_x, text_y, str(label)[:4], color='white', ha='center', va='center')        
    plt.imshow(mat, vmin=1., vmax=2,  origin='lower')
    plt.title('Diagnóstico de Gelman-Rubin')
    plt.xticks(np.arange(3), [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"])
    #plt.xticklabels([str(i) for i in [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]])
    plt.yticks(np.arange(3), ["MH", "HMC", "L2HMC"])
    #plt.yticklabels([str(i) for i in ["MH", "HMC", "HMC*"]])
    #fig.text(0.5, 0.75, '{} {}'.format(dataset, name), ha='center')
    #fig.text(0.5, 0.23, 'lambda', ha='center')
    #fig.text(0.04, 0.5, 'n', va='center', rotation='vertical')
    #plt.colorbar()
    plt.savefig(savename, dpi=dpi)

    return

"""
chains_l2hmc = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_l2hmc.npy")
chains_hmc = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_hmc.npy")
chains_mh = np.load("/home/mauricio/Documents/Uni/Intro_2/res_inf/chain_points_mh.npy")

chains_l2hmc_auto = chains_l2hmc[:, [0, 1, 2, 3, 4, 5, 6, 11, 12, 14, 15, 16, 17, 18], :]
chains_hmc_auto = chains_hmc[:, [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
chains_mh_auto = chains_mh

n, m, dim = chains_l2hmc_auto.shape
tmh = theta_m_hat(chains_l2hmc_auto)
smh = sigma_m_hat(chains_l2hmc_auto)
th = theta_hat(tmh)
B = b(tmh, th, n, m)
W = w(smh)
vh = v_hat(W, B, n, m)
dc = dcorr(vh)
rc_l2hmc = psrf(vh, W)
rc_l2hmc_d = psrf(vh, W, dc, True)

n, m, dim = chains_hmc_auto.shape
tmh = theta_m_hat(chains_hmc_auto)
smh = sigma_m_hat(chains_hmc_auto)
th = theta_hat(tmh)
B = b(tmh, th, n, m)
W = w(smh)
vh = v_hat(W, B, n, m)
dc = dcorr(vh)
rc_hmc = psrf(vh, W)
rc_hmc_d = psrf(vh, W, dc, True)

n, m, dim = chains_l2hmc_auto.shape
tmh = theta_m_hat(chains_mh_auto)
smh = sigma_m_hat(chains_mh_auto)
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
plot(R, "gelmanrubin")
plt.clf()
plot(Rd, "gelmanrubind")
"""