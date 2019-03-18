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
import pdb

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
    EE = 100.*h*torch.sqrt(arg)
    return EE, arg


def modelo(theta, z, dz, zdim):
    clight = 299792.458
    om0 = theta[:, 0]
    ol = theta[:, 1]
    w = theta[:, 2]
    dl = torch.zeros((theta.shape[0], zdim), device=ddevice)
    ok = (1 - om0 - ol).float()
    E, _ = EHubble(om0, ol, w, z, zdim)
    I = torch.cumsum(dz/(E + 1e-300), dim=1)
    o_k_s = torch.sqrt(torch.abs(ok)).view(-1, 1).repeat(1, zdim).float()
    argsin = o_k_s*I
    f1 = clight*(1 + z)*I
    f2 = clight*(1 + z)*torch.sinh(argsin)/(o_k_s + 1e-300)
    f3 = clight*(1 + z)*torch.sin(argsin)/(o_k_s + 1e-300)
    """
    28-11-2018 bug founded. model mu_obs is 43. above real data.
    """
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
	def __init__(self, dat, sigma):
		self.constant = torch.sum(-0.5*torch.log(torch.diag(2*np.pi*sigma**2, )))
		self.data = dat
		self.flat_sigma_2 = 1/torch.diag(sigma)**2
		self.flat_sigma_2_sum = torch.sum(self.flat_sigma_2)

	def get_likelihood(self, mod):
		return - 0.5*chi2(mod, self.data, self.flat_sigma_2, self.flat_sigma_2_sum)[0] + self.constant


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
    zc = z.cpu().numpy()
    zc = np.insert(zc, 0, 0)
    self.dz = torch.tensor(zc[1:] - zc[:-1], device=ddevice)
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
    theta_grad = torch.tensor(theta.clone(), requires_grad=True, device=ddevice)

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
        self.pdf = torch.ones(batch_size, dtype=torch.float, device=ddevice)/vol
        self.logpdf = torch.log(self.pdf)
    elif dist=='normal':
        self.mu = mean
        self.u = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov, device=ddevice)
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
        return  torch.tensor(np.random.uniform(low=self.low.numpy(), high=self.high.numpy(), size=(n, low.shape[0])), device=ddevice).float()


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


def acceptance_prob(state0, state1, ufunc, minf=None, msup=None):
    """
    Calculates acceptance probabilty.
    Inputs:
        state0: tensor (), 
    """
    _, ndim = state0.shape
    ap = torch.min(torch.zeros(batch_size, device=ddevice).float(), - ufunc.value(state1) + ufunc.value(state0))
    ap[torch.isnan(ap)] = - 1e3
    # 0 probablity for states out of the limits -1e20 for log(0)
    ap = limits(ap, state1, minf, msup)
    
    # selection of metropolis acceptance rule alpha>u
    un = torch.log(torch.rand((1, batch_size), device=ddevice))

    accepted = torch.t(ap>un).repeat(1, ndim).float()
    naccepted = 1 - accepted

    ap = torch.clamp(torch.exp(ap), 0., 1.)
    
    # accepted samples
    acc_sam = state1*accepted + state0*naccepted
    
    return ap, acc_sam, accepted


def limits(prob, pos, maskinf, masksup):
    """returns 0 (-1e20 for log(0)) probability to states out of prior bounds"""
    maskl = torch.prod((pos>=maskinf)*(pos<=masksup), dim=1).float()
    nmaskl = 1. - maskl
    return maskl*prob - nmaskl*1e3


class MLP(torch.nn.Module):
    def __init__(self, ndim, n1, n2, n3, nout):
        super(MLP, self).__init__()
        """
        ndim: int, # of dimensions
        n1: int, # of neurons of layer 1
        n2: int, # of neurons of layer 2
        n2: int, # of neurons of layer 3
        """
        # layers
        self.fc1 = torch.nn.Linear(ndim, n1)
        self.fc2 = torch.nn.Linear(n1, n2)
        self.fc3 = torch.nn.Linear(n2, n3)
        self.out = torch.nn.Linear(n1, nout)

        # activations
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))
        out = self.tanh(self.fc3(out))
        out = self.tanh(self.out(out))
        return out


def distance(x, y):
    """
    Euclidean distance between two vectors.
    Inputs:
        x, y: tensor (batch size, ndim).
    Outputs:
        tensor (batch size)
    """
    return torch.sum((x - y)**2, dim=1)



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
parser.add_argument("--cuda", type=bool, const=False, help="device", nargs="?", default=False)
args = parser.parse_args()

global ddevice
if torch.cuda.is_available() and args.cuda:
	ddevice = torch.device("cuda")
else:
	ddevice = torch.device("cpu")


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

redshift = torch.tensor(redshift, device=ddevice).float()
mu_obs = torch.tensor(mu_obs, device=ddevice).float()
cov = torch.tensor(cov, device=ddevice).float()

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
save = args.save
plott = args.plott

verboseprint = print if verbose else lambda *a, **k: None

Low = np.array([0., 0., -6.])
High = np.array([1., 1., -1/3])
low = torch.tensor(Low, device=ddevice).float()
high = torch.tensor(High, device=ddevice).float()
pri = Prior(low=low, high=high)
#pri = Prior(dist="normal", mean=torch.Tensor([0.5, 0.5, -3]), cov=torch.tensor([[1e-2, 0., 0.], [0., 1e-2, 0.], [0., 0., 1.]]))
pot = Potential(mu_obs, cov, redshift, pri)

#net = MLP(ndim, neurons, neurons, neurons, ndim)
net = MLP(ndim, neurons, neurons, neurons, int(ndim**2)).to(device=ddevice)

#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.5)

L = []
X = []
x = torch.tensor(np.random.uniform(low=low.cpu().numpy(), high=high.cpu().numpy(), size=(batch_size, ndim)), device=ddevice).float()
X.append(x.cpu().numpy())
cte = 1e-1
beta = 0.5

# LOG
"""
1)
NET: 100 neurons, 3 outputs
optimizer con reg L2 0.5
sigma = 1e-1*net**2
loss = mean(d + 1/d)

2)
NET: 100 neurons, ndim**2 outputs
optimizer con reg L2
sigma = 1e-1*net**2
loss = mean(d + 1/d + (sigma - sigmaT)**2)
"""

for j in range(Nt):    
    optimizer.zero_grad()
    #u = torch.randn((batch_size, ndim), requires_grad=False) # log 1
    u = torch.randn((batch_size, ndim, 1), requires_grad=False, device=ddevice) # log 2
    Sigma = cte*(net.forward(x).reshape((batch_size, ndim, ndim)))**2 # LOG 2
    #Sigma = cte*net.forward(x)**2 # LOG 1
    #xbar = x + Sigma*u # LOG 1
    xbar = x + Sigma.matmul(u).squeeze() # LOG 2
    d = distance(x, xbar)
    alpha, x_accepted, _ = acceptance_prob(x, xbar, pot, minf=low, msup=high)
    #loss = torch.mean((1 - alpha)/d + d/(1 - alpha + 1e-5))
    loss_sigma = torch.sum(torch.sum((Sigma - torch.transpose(Sigma, 1, 2))**2, dim=1), dim=1)
    loss_distance = torch.mean(d + 1/d)
    loss = beta*torch.mean(loss_distance) + (1 - beta)*torch.mean(loss_sigma)
    loss.backward()
    optimizer.step()
    x = x_accepted.detach()
    L.append(loss.detach().item())
    X.append(x.cpu().numpy())
    if j%100==0:
        print("Epoch {} Loss {:.2f}".format(j, loss))
    #pdb.set_trace()
X = np.array(X)

plt.clf()
plt.plot(np.log10(L))
plt.savefig("L.png", dpi=400)

plt.clf()
for i in range(batch_size):
	plt.figure(1)
	plt.scatter(X[:, i, 0], X[:, i, 1], color="navy", marker=".", alpha=0.1)
	

	plt.figure(2)
	plt.scatter(X[:, i, 0], X[:, i, 2], color="navy", marker=".", alpha=0.1)
	

	plt.figure(3)
	plt.scatter(X[:, i, 1], X[:, i, 2], color="navy", marker=".", alpha=0.1)
	

plt.figure(1)
plt.xlim([Low[0], High[0]])
plt.ylim([Low[1], High[1]])
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\Omega_{\Lambda}$")
plt.savefig("1.png")

plt.figure(2)
plt.xlim([Low[0], High[0]])
plt.ylim([Low[2], High[2]])
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$\omega$")
plt.savefig("2.png")

plt.figure(3)
plt.xlim([Low[1], High[1]])
plt.ylim([Low[2], High[2]])
plt.xlabel(r"$\Omega_{\Lambda}$")
plt.ylabel(r"$\omega$")
plt.savefig("3.png")
