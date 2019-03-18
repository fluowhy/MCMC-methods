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

global batch_size
global dpi

batch_size = 10
dpi = 200


def f1(theta, z, omk):
    zc = z.numpy()
    zc = np.insert(zc, 0, 0)
    dz = torch.tensor(zc[1:] - zc[:-1])
    E, _ = EHubble(theta, z)
    I = torch.cumsum(dz/(E + 1e-300), dim=1)
    o_k_s = torch.sqrt(torch.abs(omk)).view(-1, 1).repeat(1, z.shape[0]).float()
    return (1 + z)*torch.sinh(o_k_s*I)/(o_k_s + 1e-300)


def f2(theta, z, omk):
    zc = z.numpy()
    zc = np.insert(zc, 0, 0)
    dz = torch.tensor(zc[1:] - zc[:-1])
    E, _ = EHubble(theta, z)
    I = torch.cumsum(dz/(E + 1e-300), dim=1)
    o_k_s = torch.sqrt(torch.abs(omk)).view(-1, 1).repeat(1, z.shape[0]).float()
    return (1 + z)*torch.sin(o_k_s*I)/(o_k_s + 1e-300)


def f3(theta, z, omk):
    zc = z.numpy()
    zc = np.insert(zc, 0, 0)
    dz = torch.tensor(zc[1:] - zc[:-1])
    E, _ = EHubble(theta, z)
    I = torch.cumsum(dz/(E + 1e-300), dim=1)
    return (1 + z)*I


def EHubble(theta, z): # parametro de hubble
    """
    theta: parameter space state.
    z: redshift.
    bs: batch size.
    """
    zdim = z.shape[0]
    om0 = theta[:, 0].view(-1, 1).repeat(1, zdim).float()
    ol = theta[:, 1].view(-1, 1).repeat(1, zdim).float()
    w = theta[:, 2].view(-1, 1).repeat(1, zdim).float()
    arg = om0*(1 + z)**3  + (1 - om0 - ol)*(1 + z)**2 + ol*(1 + z)**(3*(1 + w))
    EE = torch.sqrt(arg)
    return EE, arg


def modelo(theta, z):    
    om0 = theta[:, 0]
    ol = theta[:, 1]
    w = theta[:, 2]
    zdim = z.shape[0]
    omega_k = (1 - om0 - ol).float()
    sig = torch.sign(omega_k).float()
    may = (1 + torch.sign(sig - 1)).view(-1, 1).repeat(1, zdim).float()
    men = (1 - torch.abs(sig)).view(-1, 1).repeat(1, zdim).float()
    eq = (1 - torch.sign(sig + 1)).view(-1, 1).repeat(1, zdim).float()
    dl = may*f1(theta, z, omega_k) + eq*f3(theta, z, omega_k) + men*f2(theta, z, omega_k)
    # integral
    dist = 5*torch.log(dl + 1e-300)/np.log(10)
    return dist


def likelihood(mod, dat, sigma):
  """Log likelihood.
  mod: tensor (batch_size, ndata): model values
  dat: array (ndata): data values
  cov: array (ndata, ndata): data error
  """
  like = - 0.5*chi2(mod, dat, sigma)[0] + torch.sum(-0.5*torch.log(torch.diag(2*np.pi*sigma**2)))
  return like


def chi2(mod, dat, cov):
  """
  mod: tensor (batch_size, ndata): model values
  dat: array (ndata): data values
  cov: array (ndata, ndata): data error
  """
  cov_p = torch.diag(cov)  
  AA = torch.sum(((dat - mod)/cov_p)**2, dim=1)
  BB = torch.sum((dat - mod)/cov_p**2, dim=1)
  CC = torch.sum(1/cov_p**2)
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


  def value(self, theta):
    """Returns potential log value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    mod = modelo(theta, self.z)
    self.u = - likelihood(mod, self.data, self.cov) - self.prior.get_log_pdf(theta)
    return self.u


  def gradi(self, theta):
    """Returns gradient value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    theta_grad = torch.tensor(theta.clone(), requires_grad=True)
    self.value(theta_grad)
    self.gradient, = torch.autograd.grad(self.u, theta_grad, grad_outputs=torch.ones(theta.shape[0]), create_graph=True)  
    return self.gradient


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
        self.u = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)

    
  def get_samples(self, n):
    """Get distribution samples.
    n: int, # of samples, batch size.
    """
    return  torch.tensor(np.random.uniform(low=low.numpy(), high=high.numpy(), size=(n, low.shape[0]))).float()


  def get_pdf(self, value):
    """Get distribution pdf.
    value: tensor, point to be evaluated.
    """
    if self.dist==None:
      return self.pdf
    else:
      return self.u.prob(value)


  def get_log_pdf(self, value):
    """Get distribution log pdf.
    value: tensor, point to be evaluated.
    """
    if self.dist==None:
      return self.logpdf
    else:
      return self.u.log_prob(value)


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
        #epsi = self.sigmoid(self.epsi(out))*1e-2
        return S, Q, T #, epsi


class Leapfrog:
    """Leapfrog object, solve dynamics."""
    def __init__(self, U, m, ndim, nnx, nnv, e, b, reg, sc, lr):
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
        self.pi = torch.tensor(self.U.prior.get_samples(batch_size)).float()
        self.msup = self.U.prior.high.repeat((batch_size, 1))
        self.minf = self.U.prior.low.repeat((batch_size, 1))
        self.reg = reg
        self.sc = sc
        self.optimizerx = torch.optim.Adam(self.nnx.parameters(), lr=lr)
        self.optimizerv = torch.optim.Adam(self.nnv.parameters(), lr=lr)
        self.learning_state = 0 # 0-train 1-evaluate

        
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
        self.Sx3, Qx, Tx = self.nnx.forward(mask*x, v, t)        
        x = x*mask + nmask*(x*torch.exp(self.e*self.Sx3) + self.e*(v*torch.exp(self.e*Qx) + Tx))        
        gradient = self.U.gradi(x)
        self.Sv4, Qv, Tv = self.nnv.forward(x, gradient, t)        
        v = v*torch.exp(0.5*self.Sv4*self.e) - 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)
        return x
        
        
    def back_dyn_fun(self, x, v, t, mask, nmask):
        """One step of backward dynamics d=-1"""
        gradient = self.U.gradi(x)
        self.Sv1, Qv, Tv = self.nnv.forward(x, gradient, t)
        v = v*torch.exp(- 0.5*self.Sv1*self.e) + 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)        
        self.Sx2, Qx, Tx = self.nnx.forward(mask*x, v, t)
        x = x*mask + nmask*(x*torch.exp(- self.e*self.Sx2) - self.e*(v*torch.exp(self.e*Qx) + Tx))        
        self.Sx3, Qx, Tx = self.nnx.forward(nmask*x, v, t)
        x = x*nmask + mask*(x*torch.exp(- self.e*self.Sx3) - self.e*(v*torch.exp(self.e*Qx) + Tx))        
        gradient = self.U.gradi(x)        
        self.Sv4, Qv, Tv = self.nnv.forward(x, gradient, t)        
        v = v*torch.exp(- 0.5*self.Sv4*self.e) + 0.5*self.e*(gradient*torch.exp(self.e*Qv) + Tv)
        return x

        
    def dyn(self, x):
        """m steps dynamics"""
        vel = self.resample()
        jacobian = 0
        for i in range(self.m):
            mask = torch.tensor(np.random.choice([1., 0.], size=(self.b, self.ndim), p=[0.5, 0.5])).float()
            nmask = 1 - mask
            # update outputs of the neural network
            t = self.time_encoding(i) # updates time
            if self.d==1:
                x = self.for_dyn_fun(x, vel, t, mask, nmask)
            elif self.d==-1:
                x = self.back_dyn_fun(x, vel, t, mask, nmask)
            jacobian += self.e*(0.5*(self.Sv1 + self.Sv4) + mask*self.Sx2 + nmask*self.Sx3)
        jacobian = torch.sum(jacobian*self.d, dim=1)
        self.flip()
        return x, jacobian

     
    def get_value(self):
        return self.x
                  
                
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
    	qi = torch.tensor(self.U.prior.get_samples(batch_size)).float()
    	qnext, jacq = self.dyn(qi)
    	pnext, jacp = self.dyn(self.pi)
    	#self.acceptance_prob()
    	Aq, _, _ = acceptance_prob(qi, qnext, jacq, self.U, self.minf, self.msup)
    	Ap, ASam, accepted = acceptance_prob(self.pi, pnext, jacp, self.U, self.minf, self.msup)
    	self.nnx.zero_grad()
    	self.nnv.zero_grad()
    	#L = self.loss()
    	L = loss(Ap, Aq, self.pi, pnext, qi, qnext, self.sc, self.reg)
    	print("Loss {}".format(L))
    	self.pi = torch.tensor(ASam.detach().numpy()).float()
    	L.backward()#retain_graph=True)
    	self.optimizerx.step()
    	self.optimizerv.step()  
    	return L, accepted


    def evolve(self, initial_sample):
    	if self.learning_state==0:
    		self.pi = initial_sample
    		self.learning_state = 1
    	pnext, jacp = self.dyn(self.pi)
    	Ap, ASam, accepted = acceptance_prob(self.pi, pnext, jacp, self.U, self.minf, self.msup)    	
    	self.pi = torch.tensor(ASam.detach().numpy()).float()
    	return self.pi, accepted


def loss(pp, pq, sp1, sp2, sq1, sq2, sc, reg):
	dp = distance(sp1, sp2)
	dq = distance(sq1, sq2)
	return torch.mean(lam(sc, dp, pp)) + torch.mean(reg*lam(sc, dq, pq))


def acceptance_prob(state0, state1, jac, ufunc, minf, msup):
	"""calculates acceptance probabilty"""
	_, ndim = state0.shape
	ap = torch.min(torch.zeros(batch_size).float(), - ufunc.value(state1) + ufunc.value(state0) + jac)
	
	# 0 probablity for states out of the limits -1e20 for log(0)
	ap = limits(ap, state1, minf, msup)
	
	# selection of metropolis acceptance rule alpha>u
	un = torch.log(torch.rand((1, batch_size)))
	accepted = torch.t(ap>un).repeat(1, ndim).float()
	naccepted = 1 - accepted

	ap = torch.clamp(torch.exp(ap), 1e-20, 1)
	
	# accepted samples
	acc_sam = state1*accepted + state0*naccepted
	
	return ap, acc_sam, accepted


def limits(prob, pos, maskinf, masksup):
	"""returns 0 (-1e20 for log(0)) probability to states out of prior bounds"""
	mask = torch.prod((pos>=maskinf)*(pos<=masksup), dim=1).float()
	nmask = 1 - mask
	return mask*prob - nmask*1e20


def distance(v1, v2):
	"""euclidean distance"""
	return torch.clamp(torch.sum((v1 - v2)**2, dim=1), 1e-20, 1e20)


def lam(sc, d, a):
	"""lambda function
	sc: scale
	d: distance
	a: acceptance probability
	"""
	a = 1
	return sc**2/d/a - d*a/sc**2


"""
def acceptance_prob(self):
	# calculates acceptance probabilty
	self.ap = torch.min(torch.zeros(batch_size).float(), - self.U.value(self.pnext) + self.U.value(self.pi) + self.jacp)
	self.aq = torch.min(torch.zeros(batch_size).float(), - self.U.value(self.qnext) + self.U.value(self.qi) + self.jacq)
	
	# 0 probablity for states out of the limits -1e20 for log(0)
	self.ap = limits(self.ap, self.pnext, self.minf, self.msup)
	self.aq = limits(self.aq, self.qnext, self.minf, self.msup)
	
	# selection of metropolis acceptance rule alpha>u
	un = torch.log(torch.rand((1, batch_size)))
	accepted = torch.t(self.ap>un).repeat(1, self.ndim).float()
	naccepted = 1 - accepted

	self.ap = torch.exp(self.ap) + 1e-20
	self.aq = torch.exp(self.aq) + 1e-20

	# accepted samples
	self.acc = self.pnext*accepted + self.pi*naccepted


def loss(self):
	dp = distance(self.pnext, self.pi)
	dq = distance(self.qnext, self.qi)
	self.acceptance_prob()
	return torch.mean(lam(self.sc, dp, self.ap)) + torch.mean(self.reg*lam(self.sc, dq, self.aq))
"""

# carga de datos

direc = "gal.txt"
#direc = "/home/mauricio/Documents/Uni/otoño_2018/Intro_2/gal.txt"

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
N = 100 # 5000
lr = 1e-3
batch_size = 10
scale = 1.0
reg = 1.0
lpsteps = 8
neurons = 5

low = torch.tensor(np.array([0, 0, -6])).float()
high = torch.tensor(np.array([1, 1, -1/3])).float()
pri = Prior(low=low, high=high)
pot = Potential(mu_obs, cov, redshift, pri)

mlpx = MLP(low.shape[0], neurons, neurons, neurons, 1.0, 1.0)
mlpv = MLP(low.shape[0], neurons, neurons, neurons, 1.0, 1.0)

leapfrog = Leapfrog(U=pot, m=lpsteps, ndim=low.shape[0], nnx=mlpx, nnv=mlpv, e=1e-2, b=batch_size, reg=reg, sc=scale, lr=lr)


# Train
P = []
Losses = []
Accepted = []
P.append(leapfrog.pi.numpy())
for j in range(N):
	print("Iteration {}".format(j))
	loss_iter, accepted = leapfrog.train()
	Losses.append(loss_iter.detach().numpy())
	P.append(leapfrog.pi.numpy())
	Accepted.append((torch.sum(accepted, dim=0)[0].numpy()))
Losses = np.array(Losses)
Accepted = np.array(Accepted)
P = np.array(P)

ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
np.save("samples_train_{}".format(ts), P)
np.save("loss_train_{}".format(ts), Losses)
np.save("acceptance_rate_train_{}".format(ts), Accepted)

"""
plt.figure(1)
plt.title("train samples")
for j in range(batch_size):
	plt.scatter(P[:, j, 0], P[:, j, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])


plt.figure(2)
plt.title("train loss")
plt.xlabel("iteration")
plt.step(np.arange(0, len(Losses), 1), Losses)

plt.figure(3)
plt.title("train acceptance rate")
plt.xlabel("iteration")
plt.step(np.arange(0, len(Accepted), 1), np.cumsum(Accepted)/np.cumsum(np.arange(batch_size, N*(batch_size + 1), batch_size)))

plt.show()
"""

# Evaluation
N = 1000
P = []
Accepted = []
Losses = []
initial_p = torch.tensor(pot.prior.get_samples(batch_size)).float()
P.append(initial_p.numpy())
for j in range(N):
	print("Iteration {}".format(j))
	sample, accepted = leapfrog.evolve(initial_p)
	P.append(leapfrog.pi.numpy())
	Accepted.append((torch.sum(accepted, dim=0)[0].numpy()))
Accepted = np.array(Accepted)
P = np.array(P)


np.save("samples_{}".format(ts), P)
np.save("acceptance_rate_{}".format(ts), Accepted)

"""
plt.figure(1)
plt.title("samples")
for j in range(batch_size):
	plt.scatter(P[:, j, 0], P[:, j, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.figure(2)
plt.title("acceptance rate")
plt.xlabel("iteration")
plt.step(np.arange(0, len(Accepted), 1), np.cumsum(Accepted)/np.cumsum(np.arange(batch_size, N*(batch_size + 1), batch_size)))

plt.show()
"""