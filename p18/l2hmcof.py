# -*- coding: utf-8 -*-
# https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
# bash ~/Anaconda3-5.2.0-Linux-x86_64.sh
# ssh mromero@leftraru.nlhpc.cl -p 22
# source activate cosmo

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as sci
import time
import os
from getdist import plots, MCSamples
import torch
import datetime
import argparse
import sys

global batch_size
global dpi

dpi = 200


def EHubble(om0, ol, w, z, zdim): # parametro de hubble
    """
    theta: parameter space state.
    z: redshift.
    bs: batch size.
    """
    h = 0.67
    om0 = om0.view(-1, 1).repeat(1, zdim).float()
    ol = ol.view(-1, 1).repeat(1, zdim).float()
    w = w.view(-1, 1).repeat(1, zdim).float()
    arg = om0*(1 + z)**3  + (1 - om0 - ol)*(1 + z)**2 + ol*(1 + z)**(3*(1 + w))
    EE = 100*h*torch.sqrt(arg)
    return EE, arg


def modelo(theta, z, dz, zdim):
	clight = 299792.458
	om0 = theta[:, 0]
	ol = theta[:, 1]
	w = theta[:, 2]
	dl = torch.zeros((theta.shape[0], zdim))
	ok = (1 - om0 - ol).float()
	E, _ = EHubble(om0, ol, w, z, zdim)
	I = torch.cumsum(dz/(E + 1e-300), dim=1)
	o_k_s = torch.sqrt(torch.abs(ok)).view(-1, 1).repeat(1, zdim).float()
	argsin = o_k_s*I
	f1 = clight*(1 + z)*I
	f2 = clight*(1 + z)*torch.sinh(argsin)/(o_k_s + 1e-300)
	f3 = clight*(1 + z)*torch.sin(argsin)/(o_k_s + 1e-300)
	sensi = 1e-3
	try:
	    dl[(ok<sensi)*(ok>0)] = f1[(ok<1e-5)*(ok>0)]
	except:
	    0
	try:
	    dl[(ok>-sensi)*(ok<0)] = f1[(ok>-1e-5)*(ok<0)]
	except:
	    0
	try:
	    dl[ok>sensi] = f2[ok>sensi]
	except:
	    0
	try:
	    dl[ok<-sensi] = f3[ok<-sensi]
	except:
	    0
	return 5*torch.log(dl + 1e-300)/np.log(10)


class Likelihood:
	"""Log likelihood.
  	mod: tensor (batch_size, ndata): model values
  	dat: array (ndata): data values
  	cov: array (ndata, ndata): data error
  	"""
	def __init__(self, data, sigma):
		self.constant = torch.sum(-0.5*torch.log(torch.diag(2*np.pi*sigma**2)))
		self.data = data
		self.chi2 = Chi2(data, sigma)

	def get_likelihood(self, mod):
		return - 0.5*self.chi2.calculate(mod) + self.constant


class Chi2():
	def __init__(self, data, cov):
		self.data = data
		self.sigma_inv_2 = 1/torch.diag(cov)**2
		self.C = torch.sum(self.sigma_inv_2)


	def calculate(self, mod):
		argsum = self.data - mod
		AA = torch.sum(self.sigma_inv_2*argsum**2, dim=1)
		BB = torch.sum(self.sigma_inv_2*argsum, dim=1)
		chi2 = AA - (BB**2)/self.C
		return chi2


"""
def chi2(mod, dat, cov, CC):
	# deprecated
	mod: tensor (batch_size, ndata): model values
	dat: array (ndata): data values
	cov: array (ndata, ndata): data error
	argsum = dat - mod
	AA = torch.sum(cov*argsum**2, dim=1)
	BB = torch.sum(argsum*cov, dim=1)
	chi = AA - (BB**2)/CC
	return chi, BB/CC
"""
##### OK


class Potential:
  def __init__(self, dat, cov, z, prior):
    """Computes potential energy and its gradient.
    dat: array (ndata), data.
    sigma: array (ndata, ndata), data error
    z: array (ndata), redshift.
    prior: object, prior.
    """
    self.data = dat
    self.cov = cov
    self.z = z
    self.prior = prior
    zc = z.numpy()
    zc = np.insert(zc, 0, 0)
    self.dz = torch.tensor(zc[1:] - zc[:-1])
    self.LikeFunc = Likelihood(dat, cov)
    self.zdim = z.shape[0]


  def value(self, theta):
    """Returns potential log value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    mod = modelo(theta, self.z, self.dz, self.zdim)
    self.u = - self.LikeFunc.get_likelihood(mod) - self.prior.get_log_pdf(theta)
    return self.u


  def gradi(self, theta):
    """Returns gradient value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    theta_grad = torch.tensor(theta.clone(), requires_grad=True)

    val = self.value(theta_grad)
    gradient, = torch.autograd.grad(val, theta_grad, grad_outputs=torch.ones(theta.shape[0]), create_graph=True)
    return gradient


class Prior:
  def __init__(self, dist=None, low=None, high=None, mean=None, cov=None):
    """Defines some priors. Uniform and normal.
    dist: str, distribution name.
    low: array or list (ndim), uniform low limit.
    high: array or list (ndim), uniform high limit.
    mean: array or list (ndim), normal dist. mean.
    cov: array or list (ndim, ndim), normal dist. covariance.

    USE ONLY UNIFORM, NORMAL NOT FINISHED
    
    """
    self.low = low
    self.high = high
    self.dist = dist 
    if dist==None:
        self.u = torch.distributions.uniform.Uniform(low=low, high=high)
        vol = torch.prod(torch.abs((high - low))).float()
        self.pdf = torch.ones(batch_size, dtype=torch.float)/vol
        self.logpdf = torch.log(self.pdf)
    elif dist=='normal':
        self.mu = mean
        self.u = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov)
        k = mean.shape[0]
        self.invcov = - 0.5*torch.inverse(cov)
        self.constant = 1/torch.sqrt((2*np.pi)**k*torch.det(cov))

    
  def get_samples(self, n):
    """Get distribution samples.
    n: int, # of samples, batch size.
    """
    if self.dist=="normal":
        return self.u.sample((n, ))
    else:
        return  torch.tensor(np.random.uniform(low=low.numpy(), high=high.numpy(), size=(n, low.shape[0]))).float()


  def get_pdf(self, x):
    """Get distribution pdf.
    value: tensor, point to be evaluated.
    """
    if self.dist==None:
        return self.pdf
    else:
        sub = x - self.mu
        return torch.diag(self.constant*torch.exp((sub.mm(self.invcov)).mm(torch.transpose(sub, 0, 1))))


  def get_log_pdf(self, x):
    """Get distribution log pdf.
    value: tensor, point to be evaluated.
    """
    if self.dist==None:
      return self.logpdf
    else:
      return torch.log(self.get_pdf(x))


class MLP(torch.nn.Module):
    def __init__(self, ndim, n1, n2, n3, ls, lq):
        super(MLP, self).__init__()
        """
        ndim: int, # of dimensions
        n1: int, # of neurons of layer 1
        n2: int, # of neurons of layer 2
        ls: float, output parameter
        lq: float, output parameter
        """
        self.ndim = ndim
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.ls = ls
        self.lq = lq
        self.fc1x = torch.nn.Linear(ndim, n1)
        self.fc1v = torch.nn.Linear(ndim, n1) 
        self.fc1t = torch.nn.Linear(2, n1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n1, n2)
        self.fc3 = torch.nn.Linear(n2, n3)
        self.s = torch.nn.Linear(n3, ndim)
        self.q = torch.nn.Linear(n3, ndim)
        self.t = torch.nn.Linear(n3, ndim)
        self.epsi = torch.nn.Linear(n3, ndim)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x, v, t):
        out = self.relu(self.fc1x(x) + self.fc1v(v) + self.fc1t(t))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        S = self.tanh(self.s(out))
        Q = self.tanh(self.q(out))
        T = self.t(out)
        #epsilon = self.sigmoid(self.epsi(out))
        return S, Q, T#, epsilon*1e-2


class Leapfrog:
    """Leapfrog object, solve dynamics."""
    def __init__(self, U, m, ndim, nnx, nnv, e, b, reg, sc, lr, msup, minf):
        """
        U: potential energy function
        m: # of leapfrog steps
        ndim: # of dimensions
        nnx: neural network of x's
        nnv: neural network of v's
        e: leapfrog step parameter
        b: batch size
        """
        self.U = U
        self.m = m
        self.ndim = ndim
        self.nnx = nnx
        self.nnv = nnv
        self.e = e
        self.b = b
        self.d = np.random.choice([-1, 1])*1.0
        self.xi = torch.tensor(self.U.prior.get_samples(batch_size)).float()
        self.reg = reg
        self.sc = sc
        self.lr = lr
        self.optimizerx = torch.optim.Adam(self.nnx.parameters(), lr=lr)
        self.optimizerv = torch.optim.Adam(self.nnv.parameters(), lr=lr)
        self.learning_state = 0 # 0-train 1-evaluate
        self.chain = []
        self.train_chain = []
        self.train_chain.append(self.xi.numpy())
        self.msup = msup
        self.minf = minf
        self.beta = 1
        self.ARate = 0

        
    def direction(self):
        """Samples a new direction"""
        self.d = np.random.choice([-1, 1])*1.0
        
        
    def for_dyn_fun(self, x, v, t, mask, nmask):
        """One step of forward dynamics d=1"""
        # remember S, Q, T update in each sub iteration
        gradient = self.U.gradi(x)
        self.Sv1, Qv, Tv = self.nnv.forward(x, gradient, t)                
        v = v*torch.exp(0.5*self.Sv1*self.e) - 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)      
        self.Sx2, Qx, Tx = self.nnx.forward(nmask*x, v, t)                
        x = x*nmask + mask*(x*torch.exp(self.e*self.Sx2) + self.e*(v*torch.exp(self.e*Qx) + Tx))
        x, v = checkAndFix(x, v)        
        self.Sx3, Qx, Tx = self.nnx.forward(mask*x, v, t)        
        x = x*mask + nmask*(x*torch.exp(self.e*self.Sx3) + self.e*(v*torch.exp(self.e*Qx) + Tx))
        x, v = checkAndFix(x, v)      
        gradient = self.U.gradi(x)
        self.Sv4, Qv, Tv = self.nnv.forward(x, gradient, t)        
        v = v*torch.exp(0.5*self.Sv4*self.e) - 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)
        return x, v
        
        
    def back_dyn_fun(self, x, v, t, mask, nmask):
        """One step of backward dynamics d=-1"""
        gradient = self.U.gradi(x)
        self.Sv1, Qv, Tv = self.nnv.forward(x, gradient, t)
        v = v*torch.exp(- 0.5*self.Sv1*self.e) + 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)        
        self.Sx2, Qx, Tx = self.nnx.forward(mask*x, v, t)
        x = x*mask + nmask*(x*torch.exp(- self.e*self.Sx2) - self.e*(v*torch.exp(self.e*Qx) + Tx))
        x, v = checkAndFix(x, v)    
        self.Sx3, Qx, Tx = self.nnx.forward(nmask*x, v, t)
        x = x*nmask + mask*(x*torch.exp(- self.e*self.Sx3) - self.e*(v*torch.exp(self.e*Qx) + Tx))
        x, v = checkAndFix(x, v)
        gradient = self.U.gradi(x)        
        self.Sv4, Qv, Tv = self.nnv.forward(x, gradient, t)        
        v = v*torch.exp(- 0.5*self.Sv4*self.e) + 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)
        return x, v

        
    def dyn(self, x, v):
        """m steps dynamics"""
        jacobian = 0
        for i in range(self.m):
            mask = torch.tensor(np.random.choice([1., 0.], size=(self.b, self.ndim), p=[0.5, 0.5])).float()
            nmask = 1 - mask
            # update outputs of the neural network
            t = self.time_encoding(i) # updates time
            if self.d==1:
                x, v = self.for_dyn_fun(x, v, t, mask, nmask)
            elif self.d==-1:
                x, v = self.back_dyn_fun(x, v, t, mask, nmask)
            jacobian += 0.5*self.e*self.Sv1 + self.e*mask*self.Sx2 + self.e*nmask*self.Sx3 + 0.5*self.Sv4*self.e
        jacobian = torch.sum(jacobian*self.d, dim=1)
        self.flip()
        return x, v, jacobian


    def time_encoding(self, mi):
        """Encodes time.
        mi: int, actual leapfrog step.
        """
        arg = 2*np.pi*mi/self.m
        val = np.array([np.cos(arg), np.sin(arg)])
        val = np.tile(val, (self.b, 1))
        return torch.tensor(val).float()        
        
        
    def resample(self):
        """Resamples velocity and direction"""
        vel = torch.randn((self.b, self.ndim)).float()
        self.direction()
        return vel
        
        
    def flip(self):
        """Flip direction"""
        self.d *= -1


    def train(self):
        """One markov chain step"""
        vi = self.resample()
        xnext, vnext, jacp = self.dyn(self.xi, vi)
        Ap = AcceptanceProbability(self.xi, xnext, vi, vnext, self.U, jacp, minf=self.minf, msup=self.msup)
        ASam, accepted = MetropolisStep(self.xi, xnext, Ap)
        self.optimizerx.zero_grad()
        self.optimizerv.zero_grad()
        #L = loss(Ap, Aq, self.pi, pnext, qi, qnext, self.sc, self.reg)
        #L = loss2(self.train_chain, ASam, 1) + self.beta*(1 - torch.sum(accepted)/self.ndim/self.b)
        L = loss3(self.xi, xnext) + self.beta*(1 - torch.sum(accepted)/self.ndim/self.b)
        #print("Loss {}".format(L))
        accepted_sample = ASam.detach().numpy()
        self.ARate += accepted.detach()    	
        self.xi = torch.tensor(accepted_sample).float()
        self.train_chain.append(accepted_sample)
        L.backward()
        self.optimizerx.step()
        self.optimizerv.step()
        return L, self.ARate


    def evolve(self):
        if self.learning_state==0:
        	self.xi = torch.tensor(pot.prior.get_samples(batch_size)).float()
        	self.learning_state = 1
        	self.chain.append(self.xi.numpy())
        vi = self.resample()
        xnext, vnext, jacp = self.dyn(self.xi, vi)
        Ap = AcceptanceProbability(self.xi, xnext, vi, vnext, self.U, jacp, minf=self.minf, msup=self.msup)
        ASam, accepted = MetropolisStep(self.xi, xnext, Ap)
        accepted_sample = ASam.detach().numpy()
        self.chain.append(accepted_sample)    	
        self.xi = torch.tensor(accepted_sample).float()
        return accepted


def AcceptanceProbability(state0, state1, vel0, vel1, ufunc, jac, minf=None, msup=None):
    prob_state_0 = - ufunc.value(state0) - kinetic(vel0)
    prob_state_1 = - ufunc.value(state1) - kinetic(vel1)
    ap = torch.min(torch.zeros(batch_size).float(), prob_state_1 - prob_state_0 + jac)
    ap = limits(ap, state1, minf, msup)
    ap = torch.clamp(torch.exp(ap), 1e-3, 1)
    return ap


def MetropolisStep(state0, state1, _ap):
    """calculates acceptance probabilty
    returns:
        acc_sam (Tensor): accepted samples
        accpeted (Tensor): accepted samples mask
    """
    _, ndim = state0.shape        
    # selection of metropolis acceptance rule alpha>u
    un = torch.rand((1, batch_size)) # torch.log(torch.rand((1, batch_size)))
    accepted = torch.t(_ap>un).repeat(1, ndim).float() 
    naccepted = 1 - accepted    
    # accepted samples
    acc_sam = state1*accepted + state0*naccepted  
    return acc_sam, accepted


def loss(pp, pq, sp1, sp2, sq1, sq2, sc, reg):
    """
    pp, pq: probabilities
    sp1, sp2, sq1, sq2: samples
    sc: scale parameter
    reg: regularization parameter
    """
    dp = distance(sp1, sp2)
    dq = distance(sq1, sq2)
    return torch.mean(lam(sc, dp, pp)) + torch.mean(reg*lam(sc, dq, pq))


def loss2(chain, next_state, k):
    """lag k autocorrelation"""
    nsamples = len(chain)
    chain = torch.tensor(chain)
    newchain = torch.cat((chain, next_state.unsqueeze(0)))
    return torch.mean(torch.sum(newchain[k:, :, :]*newchain[:-k, :, :], dim=0)/(nsamples - k + 1))


def loss3(_state, _proposal):
    """lag k autocorrelation"""
    _d = torch.mean(torch.sum((_proposal - _state)**2, dim=1))
    return 1/_d


def kinetic(_v):
    return torch.sum(0.5*_v**2, dim=1)


def acceptance_prob(state0, state1, vel0, vel1, jac, ufunc, minf=None, msup=None):
    """calculates acceptance probabilty"""
    _, ndim = state0.shape
    ap = torch.min(torch.zeros(batch_size).float(), - ufunc.value(state1) + ufunc.value(state0) + jac - kinetic(vel1) + kinetic(vel0))
    # 0 probablity for states out of the limits -1e20 for log(0)
    ap = limits(ap, state1, minf, msup)    
    # selection of metropolis acceptance rule alpha>u
    un = torch.log(torch.rand((1, batch_size)))
    accepted = torch.t(ap>un).repeat(1, ndim).float()
    naccepted = 1 - accepted
    ap = torch.clamp(torch.exp(ap), 0, 1)    
    # accepted samples
    acc_sam = state1*accepted + state0*naccepted    
    return ap, acc_sam, accepted


def limits(prob, pos, maskinf, masksup):
	"""returns 0 (-1e20 for log(0)) probability to states out of prior bounds"""
	maskl = torch.prod((pos>=maskinf)*(pos<=masksup), dim=1).float()
	nmaskl = 1 - maskl
	return maskl*prob - nmaskl*1e5


def distance(v1, v2):
	"""euclidean distance"""
	return torch.sum((v1 - v2)**2, dim=1)
	

def lam(sc, d, a):
	"""lambda function
	sc: scale
	d: distance
	a: acceptance probability
	"""
	#a = 1
	return - d*a/sc**2 #sc**2/d/a - d*a/sc**2


class Chi2Evaluator():
	def __init__(self, data, cov, z):
		self.chi2 = Chi2(data, cov)
		self.z = z
		zc = z.numpy()
		zc = np.insert(zc, 0, 0)
		self.dz = torch.tensor(zc[1:] - zc[:-1])
		self.zdim = z.shape[0]


	def calculateAndPlot(self, chain, name):
		ii, jj, _ = chain.shape
		self.C =  torch.zeros((ii, jj))
		for i in range(ii):
			mod = modelo(chain[i], self.z, self.dz, self.zdim)
			self.C[i] = self.chi2.calculate(mod)
		# plot
		self.Cn = self.C.numpy()
		plt.clf()
		for j in range(jj):
			try:
				plt.plot(self.Cn[:, j])
			except:
				0
		plt.xlabel("iteration")
		plt.ylabel(r"$\chi^{2}$")
		plt.savefig("chi2_{}.png".format(name), dpi=dpi)
		return


def PostProcessing(_chain, _accepted, chi2eval, _name, _save=False, _loss=None, _plot=False):
    _accepted = np.array(_accepted)
    _P = torch.Tensor(_chain).numpy()
    _, _N, _ = _P.shape
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if _save:
        np.save("samples_{}_{}".format(_name, ts), _P)
        np.save("acceptance_rate_{}_{}".format(_name, ts), _accepted)
        if _loss!=None:
            np.save("loss_{}_{}".format(_name, ts), np.array(_loss))

    if _plot:
        chi2eval.calculateAndPlot(torch.Tensor(_P), _name)

        plt.clf()
        for i in range(_N):
        	plt.scatter(_P[:, i, 0], _P[:, i, 1], marker=".", alpha=0.1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel(r"$\Omega_{m}$")
        plt.ylabel(r"$\Omega_{\Lambda}$")
        plt.savefig("om_ol_{}.png".format(_name), dpi=dpi)

        plt.clf()
        for i in range(_N):
        	plt.scatter(_P[:, i, 0], _P[:, i, 2], marker=".", alpha=0.1)
        plt.xlim([0, 1])
        plt.ylim([-6, -1/3])
        plt.xlabel(r"$\Omega_{m}$")
        plt.ylabel(r"w")
        plt.savefig("om_w_{}.png".format(_name), dpi=dpi)

        plt.clf()
        for i in range(_N):
        	plt.scatter(_P[:, i, 1], _P[:, i, 2], marker=".", alpha=0.1)
        plt.xlim([0, 1])
        plt.ylim([-6, -1/3])
        plt.xlabel(r"$\Omega_{\Lambda}$")
        plt.ylabel(r"w")
        plt.savefig("ol_w_{}.png".format(_name), dpi=dpi)

        if _loss!=None:
            plt.clf()
            plt.plot(np.array(_loss), color="navy")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.savefig("loss.png", dpi=dpi)


def checkAndFix(state, velocity):
    newstate = torch.ones((batch_size, 3))
    newvelocity = torch.ones((batch_size, 3))
    mask = torch.ones((batch_size, 3))

    mask[state[:, 0]<0, 0] = 0
    mask[state[:, 1]<0, 1] = 0
    mask[state[:, 2]>-1./3, 2] = 0

    nmask = torch.ones((batch_size, 3)) - mask
    
    newstate[:, 0] = nmask[:, 0]*state[:, 0]*-1 + mask[:, 0]*state[:, 0]
    newstate[:, 1] = nmask[:, 1]*state[:, 1]*-1 + mask[:, 1]*state[:, 1]
    newstate[:, 2] = nmask[:, 2]*(- 1./3 - (state[:, 2] - - 1./3)) + mask[:, 2]*state[:, 2]

    newvelocity[:, 0] = nmask[:, 0]*velocity[:, 0]*-1 + mask[:, 0]*velocity[:, 0]
    newvelocity[:, 1] = nmask[:, 1]*velocity[:, 1]*-1 + mask[:, 1]*velocity[:, 1]
    newvelocity[:, 2] = nmask[:, 2]*velocity[:, 2]*-1 + mask[:, 2]*velocity[:, 2]

    return newstate, newvelocity

    """
     if qe[0]<0:
                    qe[0] = qe[0]*-1
                    pe[0] = pe[0]*-1
                if qe[1]<0:
                    qe[1] = qe[1]*-1
                    pe[1] = pe[1]*-1
                if qe[2]>1/3:
                    qe[2] = 1/3 - qe[2]
                    pe[2] = pe[2]*-1
    """


if __name__=="__main__":

    # carga de datos

    direc = "gal.txt"

    redshift = np.genfromtxt(direc, usecols=(1))
    mu_obs = np.genfromtxt(direc, usecols=(2)) # m - M
    cov = np.genfromtxt(direc, usecols=(3))

    p = np.argsort(redshift)
    redshift = redshift[p].astype(np.float32)
    mu_obs = mu_obs[p]
    cov = cov[p]
    cov = np.diag(cov)

    redshift = torch.tensor(redshift).float()
    mu_obs = torch.tensor(mu_obs).float()
    cov = torch.tensor(cov).float()

    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--Nt", type=int, const=5000, help="training iterations", nargs="?", default=5000)
    parser.add_argument("--Ne", type=int, const=10000, help="evaluation iterations", nargs="?", default=10000)
    parser.add_argument("--b", type=int, const=40, help="batch size", nargs="?", default=40)
    parser.add_argument("--ls", type=int, const=8, help="leapfrog steps", nargs="?", default=8)
    parser.add_argument("--ne", type=int, const=40, help="neurons per layer", nargs="?", default=40)
    parser.add_argument("--lr", type=float, const=1e-3, help="learning rate", nargs="?", default=1e-3)
    parser.add_argument("--sc", type=float, const=1, help="scale", nargs="?", default=1)
    parser.add_argument("--r", type=float, const=1, help="regularization", nargs="?", default=1)
    parser.add_argument("--ep", type=float, const=1e-2, help="leapfrog step size", nargs="?", default=1e-2)
    parser.add_argument("--ver", type=bool, const=False, help="verbose", nargs="?", default=False)
    parser.add_argument("--sav", type=bool, const=False, help="save chains", nargs="?", default=False)
    parser.add_argument("--plott", type=bool, const=False, help="plot chains", nargs="?", default=False)
    args = parser.parse_args()

    # train parameters from paper
    Nt =args.Nt # train
    Ne = args.Ne #10000 # evaluation
    lr = args.lr
    batch_size = args.b # 40
    scale = args.sc
    reg = args.r
    lpsteps = args.ls
    neurons = args.ne # 40
    ndim = 3
    ep = args.ep
    verbose = args.ver
    save = args.sav
    plott = args.plott

    verboseprint = print if verbose else lambda *a, **k: None

    low = torch.tensor(np.array([0.01, 0.01, -6])).float()
    high = torch.tensor(np.array([1, 1, -1/3])).float()
    pri = Prior(low=low, high=high)
    #pri = Prior(dist="normal", mean=torch.Tensor([0.5, 0.5, -3]), cov=torch.tensor([[1e-2, 0., 0.], [0., 1e-2, 0.], [0., 0., 1.]]))
    pot = Potential(mu_obs, cov, redshift, pri)

    mlpx = MLP(low.shape[0], neurons, neurons, neurons, 1.0, 1.0)
    mlpv = MLP(low.shape[0], neurons, neurons, neurons, 1.0, 1.0)

    leapfrog = Leapfrog(U=pot, m=lpsteps, ndim=low.shape[0], nnx=mlpx, nnv=mlpv, e=ep, b=batch_size, reg=reg, sc=scale, lr=lr, msup=high, minf=low)

    # chi2 evaluator
    chi2 = Chi2Evaluator(mu_obs, cov, redshift)

    if plott:
    	plt.ion()
    	plt.xlim([0, 1])
    	plt.ylim([0, 1])
    # Train
    Losses = []
    Accepted = []
    ti = time.time()
    for j in range(Nt):
    	try:			
            loss_iter, accepte = leapfrog.train()
            Accepted.append(accepte.numpy())
            Losses.append(loss_iter.detach().numpy())
            if j%10==0:
                tf = time.time()
                Loss = Losses[j].item()
                verboseprint("Train iteration {} time {:.2f} Loss {:.2f} Acceptance ratio {:.2f}".format(j, tf - ti, Loss, torch.mean(accepte[:, 0])/(j + 1)))#, object(), 3)
                ti = time.time()
                if np.isnan(Loss):
                    break
            if plott:
            	point = leapfrog.train_chain[-1]
            	plt.scatter(point[:4, 0], point[:4, 1], color=['navy','green','red', "brown"], marker=".", alpha=0.1)
            	plt.pause(1e-4)
    	except KeyboardInterrupt:
            print('Interrupted')
            PostProcessing(leapfrog.train_chain, Accepted, chi2, _name="train", _save=save, _loss=Losses, _plot=True)
            break
    		#sys.exit(0)

    PostProcessing(leapfrog.train_chain, Accepted, chi2, _name="train", _save=save, _loss=Losses, _plot=True)
    #PostProcessing(Losses, Accepted, chi2, save)

    # Evaluation
   
    Accepted = []
    ti = time.time()
    for j in range(Ne):
        try:
            accepted = leapfrog.evolve()
            Accepted.append(accepte.numpy())
            if j%10==0:
                tf = time.time()
                verboseprint("Eval iteracion {} tiempo {:.2f}".format(j, tf - ti))
                ti = time.time()
        except KeyboardInterrupt:
            print('Interrupted')
            PostProcessing(leapfrog.train_chain, Accepted, chi2, _name="eval", _save=save, _loss=Losses, _plot=True)
            break
            sys.exit(0)
    PostProcessing(leapfrog.chain, Accepted, chi2, _name="eval", _save=save, _plot=True)