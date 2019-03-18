#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as sci
import time
from getdist import plots, MCSamples
import time

dpi = 200

"""
Hamiltonian Monte Carlo and Metropolis-Hastings.
"""

class LeapFrog:
    def __init__(self, l, e, dw, m, z, dat, sigma):
        self.l = l
        self.e = e
        self.dw = dw
        self.m = m
        self.x = z
        self.y = dat
        self.sigma = sigma
        self.dim = len(self.m)

        
    def solve(self, theta):        
        qe = theta
        while True:
            pi = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=np.diag(self.m))
            pe = pi
            self.H = []
            self.X = []
            self.P = []
            self.X.append(theta)
            self.P.append(pe)
            for i in range(self.l):
                pe = pe - 0.5*self.e*gradiente(self.dw, qe, self.x, self.y, self.sigma) # actualiza momento en e/2
                qe = qe + self.e*pe/self.m
                if qe[0]<0:
                    qe[0] = qe[0]*-1
                    pe[0] = pe[0]*-1
                if qe[1]<0:
                    qe[1] = qe[1]*-1
                    pe[1] = pe[1]*-1
                if qe[2]>1/3:
                    qe[2] = 1/3 - qe[2]
                    pe[2] = pe[2]*-1
                pe = pe - 0.5*self.e*gradiente(self.dw, qe, self.x, self.y, self.sigma)
                self.P.append(pe)
                self.X.append(qe)
                if i + 1==self.l: 
                    i += 1
                self.H.append(hamiltoniano(p=pe, dat=self.y, sigma=self.sigma, theta=theta, z=self.x, m=self.m))
            if i==self.l: 
                break
    
    
    def get(self):
        return np.array(self.X), np.array(self.P), np.array(self.H)

    
def likelihood(mod, dat, sigma): # retorna escalar, log(L)
    sig = np.diagonal(sigma)
    L = -0.5*chi2(mod, dat, sigma)[0]  + np.sum(-0.5*np.log(2*np.pi*sig**2))
    #pp = np.argwhere((a1==-np.inf))
    #a1[pp] = 0
    return L


def chi2(mod, dat, sigma):
    sig = np.diagonal(sigma)
    AA = np.sum(((dat - mod)/sig)**2)
    BB = np.sum((dat - mod)/sig**2)	
    CC = np.sum(1/sig**2)
    chi = AA - (BB**2)/CC
    return chi, BB/CC


def prior(theta): # log(pi)
    ct = 1
    r = np.diag(np.ones(len(theta))*ct)
    p = -0.5*np.log(np.linalg.det(2*np.pi*r)) - 0.5*theta.dot((np.linalg.inv(r)).dot(theta)) 
    return p


def acepta_hmc(ec, ep, EC, EP, x, X):
    alpha = min(- EP - EC + ep + ec, 0) # log(alpha)
    u = np.log(np.random.uniform())
    if u<alpha:
        return X, EP
    else:
        return x, ep
    
    
def acepta_mh(T1, pos1, T2, pos2, m1, m2):
    alpha = min(pos2 - pos1, 0) # log(alpha)
    u = np.log(np.random.uniform())
    if u<alpha:
        return T2, pos2, m2
    else:
        return T1, pos1, m1    


def EHubble(theta, z): # parametro de hubble
    om0 = theta[0]
    ol = theta[1]
    w = theta[2]
    arg = om0*(1 + z)**3 + (1 - om0 - ol)*(1 + z)**2 + ol*(1 + z)**(3*(1 + w))
    EE = np.sqrt(arg)
    return EE, arg


def modelo(theta, z): # modulo de la distancia teorico
    """
    theta: parameter space vector
    z: redshift
    """
    om0 = theta[0]
    ol = theta[1]
    omega_k = 1 - om0 - ol
    E = EHubble(theta, z)[0]
    I = sci.cumtrapz(1/(E + 1e-300), z, initial=0)+z[0]*((1/(E + 1e-300))[0] + 1)/2 # estabilidad numerica
    o_k_s = np.sqrt(abs(omega_k))
    if omega_k==0:
        dl = (1 + z)*I
    elif omega_k<0:
        dl = (1 + z)*np.sin(o_k_s*I)/(o_k_s + 1e-300) # estabilidad numerica
    elif omega_k>0:	
        dl = (1 + z)*np.sinh(o_k_s*I)/(o_k_s + 1e-300) # estabilidad numerica
    dist = 5*np.log10(dl + 1e-300) # estabilidad numerica
    #f (-np.inf==dist).any(): 
    #    print(theta)
    return dist


def tasa(tant, tpos):
    l = len(tant)
    if np.sum(tant==tpos)==l:
        c = 0
    else:
        c = 1
    return c


def revisa(theta, z):
    arg = EHubble(theta, z)[1]
    bol = np.sum(arg<0)
    print(bol)
    if bol>0:
        a = 0 # raiz imaginaria 
    else:
        a = 1 # raiz real
    return a


def argmin2(t1, t2, t3, V): # busca en base a vector chi2 y devulve minimos de los parametros
    amin = np.argmin(V)
    return t1[amin], t2[amin], t3[amin]


def revisa1(X):
    x = X[0]
    y = X[1]
    z = X[2]
    xlim = np.array([0, 1])
    ylim = xlim
    zlim = np.array([-np.inf, 1/3])
    if xlim[0]<x<xlim[1] and ylim[0]<y<ylim[1] and zlim[0]<z<zlim[1]:
        return 1
    else:
        return 0


def potencial(dat, sigma, theta, z):
    mod = modelo(theta, z)
    u = - likelihood(mod, dat, sigma) - prior(theta) 
    return u


def cinetica(p, m):
    k = np.sum(p**2/2/m)
    return k


def hamiltoniano(p, dat, sigma, theta, z, m=1):
    h = cinetica(p, m) + potencial(dat, sigma, theta, z)
    return h


def gradiente(dw, theta, z, dat, sigma):
    tf = theta + dw
    tb = theta - dw
    grad = (potencial(dat, sigma, tf, z) - potencial(dat, sigma, tb, z))/(2*dw)
    return grad


def HMC(modelo, datos, ds, dg, N, L, params, q0, cov_mod, m, des=0.24):
    """
    datos: X, F(X)
    params: ['p1', 'p2', ..., 'pn']
    cov: matriz de covarianza de datos
    """
    # Matrices de datos de la cadena
    X = datos[0]
    Y = datos[1]
    chain = [] 
    post = [] 
    chi_2 = []
    Ratio = []
    acept = 0
    mod1 = modelo(q0, X)
    pos1 = potencial(Y, cov_mod, q0, X)
    Chi1 = chi2(mod1, Y, cov_mod)[0]
    chain.append(q0)
    post.append(pos1)
    chi_2.append(Chi1)
    Ratio.append(100)
    
    leap = LeapFrog(l=L, e=ds, dw=dg, m=m, z=X, dat=Y, sigma=cov_mod)

    Ti = time.time()
    for i in range(N):
        q = chain[i]
        while True:
            leap.solve(q)
            Q, P, H = leap.get()
            Q1 = Q[-1]
            P1 = P[-1]
            if revisa1(Q1):
                break
        t = cinetica(P1[0], m)
        u = potencial(Y, cov_mod, q, X)
        T = cinetica(P1, m)
        U = potencial(Y, cov_mod, Q1, X)
        A = acepta_hmc(t, u, T, U, q, Q1)
        chain.append(A[0])
        post.append(A[1])
        mod1 = modelo(A[0], X)
        Chi1 = chi2(mod1, Y, cov_mod)[0]
        chi_2.append(Chi1)
        # ratio de aceptacion
        acept += tasa(chain[i], chain[i + 1]) 
        Ratio.append(acept/(i+1)*100)
        if i%100==0:
            print("{}/{}".format(i, N))
            print("ratio {}%".format(Ratio[i]))

    Tf = time.time()
    print("Total time elapsed: {} s".format(np.around(Tf - Ti, 0)))

    post = np.array(post)
    chain = np.array(chain)
    chi_2 = np.array(chi_2)
    Ratio = np.array(Ratio)
    H = np.array(H)

    t1 = chain[:,0]
    t2 = chain[:,1]
    t3 = chain[:,2]

    # busca argumento del minimo de chi2
    t1m, t2m, t3m = np.around(argmin2(t1, t2, t3, chi_2),3)
    mins = [t1m, t2m, t3m]
    muestras = {}
    for i in range(len(params)):
        muestras[params[i]] = chain[:, i]
    return muestras, Ratio, chi_2, post, mins


def MH(modelo, datos, N, params, q0, cov_mod, cov_prop, des=0.24):
    """
    datos: X, F(X)
    params: ['p1', 'p2', ..., 'pn']
    cov: matriz de covarianza de datos
    """
    # Matrices de datos de la cadena
    #pid = PID(kp=10, ki=10, kd=10, o_min=2, o_max=20)
    T0 = q0
    X = datos[0]
    Y = datos[1]
    chain = [] 
    post = [] 
    chi_2 = []
    Ratio = []
    mod = []
    acept = 0
    mod0 = modelo(T0, X)
    chi0 = chi2(mod0, Y, cov_mod)[0]
    pos0 = likelihood(mod0, Y, cov_mod) + prior(T0)
    mod.append(mod0)
    chain.append(T0)
    post.append(pos0)
    chi_2.append(chi0)
    Ratio.append(100)

    # pasos de cadena
    Ti = time.time()
    for i in range(N):
        # revisa si se paso umbral de burn in
        """	
        if chi_2[i]<=580 and d==0 and o!=0:
            covarianza = COV[o]
            d = 1
            print('actualizada')
            print(covarianza)
        """	
        # selecciona ultimo elemento de la cadena
        T0 = chain[i]
        # itera hasta que encuentra un proposal valido
        while True:
            T1 = np.random.multivariate_normal(mean=T0, cov=cov_prop)
            if revisa1(T1):
                break
        # selecciona ultimo modelo
        mod0 = mod[i]
        # calcula modelo con proposal
        mod1 = modelo(T1, X)
        # selecciona ultima dis. post.
        pos0 = post[i]
        # calcula nueva dist. post.
        pos1 = likelihood(mod1, Y, cov_mod) + prior(T1)
        # decision de aceptacion
        A = acepta_mh(T0, pos0, T1, pos1, mod1, mod1)
        # guarda la variable aceptada (puede ser la anterior o proposal)
        chain.append(A[0])
        post.append(A[1])
        mod.append(A[2])
        chi_2.append(chi2(A[2], Y, cov)[0])
        # ratio de aceptacion
        acept += tasa(chain[i], chain[i + 1]) 
        Ratio.append(acept/(i+1)*100)
        if i%100==0:
            print("{}/{}".format(i, N))
            print("ratio {}%".format(Ratio[i]))

    Tf = time.time()
    print("Total time elapsed: {} s".format(np.around(Tf - Ti, 0)))
    
    ratio = acept/N*100
    print("ratio {}%".format(np.rint(ratio)))

    post = np.array(post)
    chain = np.array(chain)
    chi_2 = np.array(chi_2)
    Ratio = np.array(Ratio)
  
    t1 = chain[:,0]
    t2 = chain[:,1]
    t3 = chain[:,2]

    # busca argumento del minimo de chi2
    t1m, t2m, t3m = np.around(argmin2(t1, t2, t3, chi_2),3)
    mins = [t1m, t2m, t3m]
    muestras = {}
    for i in range(len(params)):
        muestras[params[i]] = chain[:, i]
    return muestras, Ratio, chi_2, post, mins


def plot(arr1, arr2, keys, names, save=None):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    ax1.scatter(arr1[0][keys[0]], arr1[0][keys[1]], marker='.', alpha=0.1)
    ax1.scatter(arr1[0][keys[0]][0], arr1[0][keys[1]][0], marker='x', color='red', alpha=1, label='inicio')
    ax1.scatter(arr1[4][0], arr1[4][1], marker='o', color='black', alpha=1, label='best fit')
    ax2.scatter(arr2[0][keys[0]], arr2[0][keys[1]], marker='.', alpha=0.1)
    ax2.scatter(arr2[0][keys[0]][0], arr2[0][keys[1]][0], marker='x', color='red', alpha=1, label='inicio')
    ax2.scatter(arr2[4][0], arr2[4][1], marker='o', color='black', alpha=1, label='best fit')
    ax1.set_title('HMC')
    ax2.set_title('MH')
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel(names[0])
    ax2.set_ylabel(names[1])
    ax1.set_ylabel(names[1])
    if save!=None:
        fig.savefig('muestras_'+keys[0]+'_'+keys[1], dpi=dpi)
    return


def autocorrelation(x, kmax=None):
    N = len(x)
    xmean = np.mean(x)
    den = np.sum((x - xmean)*(x - xmean))    
    if kmax==None:
        kmax = int(N/2)
    C = []
    for i in range(kmax):
        if i==0:
            corr = 1
        else:
            corr = np.sum((x[i:] - xmean)*(x[:-i] - xmean))/den
        C.append(corr)
    return np.array(C)


if __name__=="__main__":
    # saved things directory
    # path of saved data
    direc = "/home/claudia/Documents/Mau/Uni/primavera_2018/cosmo/gal.txt"
    # path of results folder
    savepath = "/home/claudia/Documents/Mau/Uni/primavera_2018/cosmo/code"

    # data loader
    redshift = np.genfromtxt(direc, usecols=(1))
    mu_obs = np.genfromtxt(direc, usecols=(2)) # m - M
    cov = np.genfromtxt(direc, usecols=(3))

    p = np.argsort(redshift)
    redshift = redshift[p]
    mu_obs = mu_obs[p]
    cov = cov[p]
    cov = np.diag(cov)

    # initial sample, bounds and definitions
    low = [0., 0., -4.] # low bound
    high = [1., 1., -1/3] # high bound
    cov_ini = np.diag(np.array([0.4, 1.125, 4])**2)*0.5e-2 # initial covariance
    q0 = np.random.uniform(low=low, high=high) # initial sample
    mode = modelo(q0, redshift) # model to use
    chi = chi2(mode, mu_obs, cov) # chi function
    N = 50000 # number of iterations

    # chain configuration

    # params
    labs = [r'$\Omega_{m}$', r'$\Omega_{\Lambda}$', r'w']
    labs1 = [r'\Omega_{m}', r'\Omega_{\Lambda}', r'w']

    M = -19.3182761161

    """
    Single chain Metropolis Hastings
    """
    print("Metropolis-Hastings")
    q0 = np.random.uniform(low=low, high=high) # np.array([0.5, 0.4, -2])
    R_mh = MH(modelo, [redshift, mu_obs], N=N, params=['om', 'ol', 'w'], q0=q0, cov_mod=cov, cov_prop=cov_ini)
    chain = np.vstack((R_mh[0]['om'], R_mh[0]['ol'], R_mh[0]['w'])).T

    np.save("{}/samples_mh".format(savepath), chain) # save with shape npoints, nchain, nparameters

    """
    Single chain Hamiltonian Monte Carlo
    """
    """
    ds = 1e-2 # leapfrog solver step size
    m = np.array([1, 1, 1]) # particle mass
    L = 8 # leapfrog solver steps number, select between 5 and 10 steps
    dg = 1e-6 # numerical gradient step size    

    print("Hamiltonian Monte Carlo")
    R_hmc = HMC(modelo, [redshift, mu_obs], ds=ds, dg=dg, N=N, L=L, params=['om', 'ol', 'w'], q0=q0, cov_mod=cov, m=m)
    chain = np.vstack((R_hmc[0]['om'], R_hmc[0]['ol'], R_hmc[0]['w'])).T

    np.save("{}/samples_hmc".format(savepath), chain)
    """
    
    """
    Plotting
    """

    # Samples

    names = [r'$\Omega_{m}$', r'$\Omega_{\Lambda}$']
    keys = ['om', 'ol']
    plot(R_hmc, R_mh, keys, names, save='yes')

    names = [r'$\Omega_{m}$', r'$\omega$']
    keys = ['om', 'w']
    plot(R_hmc, R_mh, keys, names, save='yes')

    names = [r'$\Omega_{\Lambda}$', r'$\omega$']
    keys = ['ol', 'w']
    plot(R_hmc, R_mh, keys, names, save='yes please')

    # Distributions

    ndim = 3
    names = ["x%s"%i for i in range(ndim)]
    labels = ["x_%s"%i for i in range(ndim)]
    t1 = R_hmc[0]['om']
    t2 = R_hmc[0]['ol']
    t3 = R_hmc[0]['w']
    samps = np.vstack((t1, t2, t3)).T
    samples = MCSamples(samples=samps, names=labs1, labels=labs1)

    # Triangle plot
    plt.clf()
    g = plots.getSubplotPlotter()
    samples.updateSettings({'contours': [0.68, 0.95, 0.99]})
    g.settings.num_plot_contours = 3
    g.triangle_plot([samples], filled=True)
    plt.savefig("{}/dist".format(savepath), dpi=dpi)

    # autocorrelation

    ac_hmc_om = autocorrelation(R_hmc[0]['om'])
    ac_mh_om = autocorrelation(R_mh[0]['om'])
    
    plt.clf()
    plt.plot(ac_hmc_om, label='HMC')
    plt.plot(ac_mh_om, label='MH')
    plt.title('autocorrelation')
    plt.legend()
    plt.xlabel('k')
    plt.savefig("{}/autocorrelation".format(savepath), dpi=dpi)
    