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

global batch_size
global dpi

dpi = 200


def EHubble(om0, ol, w, z, zdim): # parametro de hubble
    """
    theta: parameter space state.
    z: redshift.
    bs: batch size.
    """
    om0 = om0.view(-1, 1).repeat(1, zdim).float()
    ol = ol.view(-1, 1).repeat(1, zdim).float()
    w = w.view(-1, 1).repeat(1, zdim).float()
    arg = om0*(1 + z)**3  + (1 - om0 - ol)*(1 + z)**2 + ol*(1 + z)**(3*(1 + w))
    EE = torch.sqrt(arg)
    return EE, arg


def modelo(theta, z, dz, zdim):
    om0 = theta[:, 0]
    ol = theta[:, 1]
    w = theta[:, 2]
    dl = torch.zeros((theta.shape[0], zdim))
    ok = (1 - om0 - ol).float()
    E, _ = EHubble(om0, ol, w, z, zdim)
    I = torch.cumsum(dz/(E + 1e-300), dim=1)
    o_k_s = torch.sqrt(torch.abs(ok)).view(-1, 1).repeat(1, zdim).float()
    argsin = o_k_s*I
    f1 = (1 + z)*I
    f2 = (1 + z)*torch.sinh(argsin)/(o_k_s + 1e-300)
    f3 = (1 + z)*torch.sin(argsin)/(o_k_s + 1e-300)
    try:
    	dl[ok==0] = f1[ok==0]
    except:
    	0
    try:
    	dl[ok>0] = f2[ok>0]

    except:
        0
    try:
    	dl[ok<0] = f3[ok<0]        
    except:
        0
    return 5*torch.log(dl + 1e-300)/np.log(10)


class Likelihood:
	"""Log likelihood.
  	mod: tensor (batch_size, ndata): model values
  	dat: array (ndata): data values
  	cov: array (ndata, ndata): data error
  	"""
	def __init__(self, dat, sigma):
		self.constant = torch.sum(-0.5*torch.log(torch.diag(2*np.pi*sigma**2)))
		self.data = dat
		self.flat_sigma_2 = 1/torch.diag(sigma)**2
		self.flat_sigma_2_sum = torch.sum(self.flat_sigma_2)

	def get_likelihood(self, mod):
		return - 0.5*chi2(mod, self.data, self.flat_sigma_2, self.flat_sigma_2_sum)[0] + self.constant


class LikelihoodLinearModel():
	def __init__(self):
		pass

	def get_likelihood(self):
		pass


def chi2(mod, dat, cov, CC):
  """
  mod: tensor (batch_size, ndata): model values
  dat: array (ndata): data values
  cov: array (ndata, ndata): data error
  """
  argsum = dat - mod
  AA = torch.sum(cov*argsum**2, dim=1)
  BB = torch.sum(argsum*cov, dim=1)
  chi = AA - (BB**2)/CC
  return chi, BB/CC

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


class Leapfrog:
    """Leapfrog object, solve dynamics."""
    def __init__(self, U, m, ndim, e, b, msup, minf):
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
        self.e = e
        self.b = b
        self.xi = torch.tensor(self.U.prior.get_samples(batch_size)).float()
        self.chain = []
        self.chain.append(self.xi.numpy())
        self.msup = msup
        self.minf = minf
        self.ARate = 0

        
    def dynamics(self, _x, _v):
        """One step of forward dynamics d=1"""
        # remember S, Q, T update in each sub iteration
        _gradient = self.U.gradi(_x)        
        _v = _v - 0.5*self.e*_gradient
        _x = _x + self.e*_v
        _gradient = self.U.gradi(_x)
        _v = _v - 0.5*self.e*_gradient
        return _x, _v

        
    def dyn(self, x):
        """m steps dynamics"""
        self.vi = torch.randn((self.b, self.ndim)).float()
        for i in range(self.m):
            x, v = self.dynamics(x, self.vi)
        return x, v


    def evolve(self):
        xnext, vnext = self.dyn(self.xi)
        Ap, ASam, accepted = acceptance_prob(self.xi, xnext, self.vi, vnext, self.U, self.minf, self.msup)
        self.ARate += accepted.detach()
        accepted_sample = ASam.detach().numpy()
        self.chain.append(accepted_sample)
        self.xi = torch.tensor(accepted_sample).float()
        return self.ARate


def kinetic(_v):
    return torch.sum(0.5*_v**2, dim=1)


def acceptance_prob(state0, state1, vel0, vel1, ufunc, minf=None, msup=None):
    """calculates acceptance probabilty"""
    _, ndim = state0.shape
    ap = torch.min(torch.zeros(batch_size).float(), - ufunc.value(state1) + ufunc.value(state0) - kinetic(vel1) + kinetic(vel0))
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
	return maskl*prob - nmaskl*1e20


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
parser.add_argument("--save", type=bool, const=True, help="save chains", nargs="?", default=True)
parser.add_argument("--plott", type=bool, const=False, help="plot chains", nargs="?", default=False)
args = parser.parse_args()

# carga de datos

direc = "/home/claudia/Documents/Mau/Uni/primavera_2018/cosmo/gal.txt"

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

# train parameters from paper
Nt = args.Nt # train
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
save = args.save
plott = args.plott

verboseprint = print if verbose else lambda *a, **k: None

low = torch.tensor(np.array([0.01, 0.01, -6])).float()
high = torch.tensor(np.array([1, 1, -1/3])).float()
pri = Prior(low=low, high=high)
#pri = Prior(dist="normal", mean=torch.Tensor([0.5, 0.5, -3]), cov=torch.tensor([[1e-2, 0., 0.], [0., 1e-2, 0.], [0., 0., 1.]]))
pot = Potential(mu_obs, cov, redshift, pri)
leapfrog = Leapfrog(U=pot, m=lpsteps, ndim=low.shape[0], e=ep, b=batch_size, msup=high, minf=low)

if plott:
	plt.ion()
	plt.xlim([0, 1])
	plt.ylim([0, 1])

Accepted = []
for j in range(Nt):
	ti = time.time()
	accepte = leapfrog.evolve()
	Accepted.append(accepte.numpy())
	tf = time.time()
	verboseprint("Iteration {} time {:.2f} Acceptance ratio {:.2f}".format(j, tf - ti, torch.mean(accepte[:, 0])/j))#, object(), 3)
	if plott:
		point = leapfrog.chain[-1]
		plt.scatter(point[:4, 0], point[:4, 1], color=['navy','green','red', "brown"], marker=".", alpha=0.1)
		plt.pause(1e-4)
Accepted = np.array(Accepted)
P = np.array(leapfrog.chain)
ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if save:
    np.save("samples_hmc_{}".format(ts), P)
    np.save("acceptance_rate_hmc_{}".format(ts), Accepted)

plt.clf()
plt.scatter(P[:, :5, 0], P[:, :5, 1], color=["navy","green","red", "brown", "purple"], marker=".", alpha=0.1)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\Omega_{\Lambda}$")
plt.savefig("hmc1.png", dpi=dpi)

plt.clf()
plt.scatter(P[:, :5, 0], P[:, :5, 2], color=["navy","green","red", "brown", "purple"], marker=".", alpha=0.1)
plt.xlim([0, 1])
plt.ylim([-6, -1/3])
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"w")
plt.savefig("hmc2.png", dpi=dpi)

plt.clf()
plt.scatter(P[:, :5, 1], P[:, :5, 2], color=["navy","green","red", "brown", "purple"], marker=".", alpha=0.1)
plt.xlim([0, 1])
plt.ylim([-6, -1/3])
plt.xlabel(r"$\Omega_{\Lambda}$")
plt.ylabel(r"w")
plt.savefig("hmc3.png", dpi=dpi)