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


def model(_theta, _x):
	xdim = len(_x)
	_m = _theta[:, 0].view(-1, 1).repeat(1, xdim)
	_b = _theta[:, 1].view(-1, 1).repeat(1, xdim)
	return _m*_x + _b





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
        return S, Q, T #, epsilon*1e-2


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
        self.reg = reg
        self.sc = sc
        self.lr = lr
        self.optimizerx = torch.optim.Adam(self.nnx.parameters(), lr=lr)
        self.optimizerv = torch.optim.Adam(self.nnv.parameters(), lr=lr)
        self.learning_state = 0 # 0-train 1-evaluate
        self.chain = []
        self.train_chain = []
        self.train_chain.append(self.pi.numpy())
        if self.U.prior.dist==None:
        	self.msup = self.U.prior.high.repeat((batch_size, 1))
        	self.minf = self.U.prior.low.repeat((batch_size, 1))
        else:
        	self.msup = None
        	self.minf = None

        
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
        return x, v
        
        
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
        return x, v

        
    def dyn(self, x):
        """m steps dynamics"""
        self.vi = self.resample()
        jacobian = 0
        for i in range(self.m):
            mask = torch.tensor(np.random.choice([1., 0.], size=(self.b, self.ndim), p=[0.5, 0.5])).float()
            nmask = 1 - mask
            # update outputs of the neural network
            t = self.time_encoding(i) # updates time
            if self.d==1:
                x, v = self.for_dyn_fun(x, self.vi, t, mask, nmask)
            elif self.d==-1:
                x, v = self.back_dyn_fun(x, self.vi, t, mask, nmask)
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
    	#qi = torch.tensor(self.U.prior.get_samples(batch_size)).float()
    	#qnext, jacq = self.dyn(qi)
    	pnext, vnext, jacp = self.dyn(self.pi)
    	#Aq, _, _ = acceptance_prob(qi, qnext, jacq, self.U, self.minf, self.msup)
    	Ap, ASam, accepted = acceptance_prob(self.pi, pnext, self.vi, vnext, jacp, self.U, self.minf, self.msup)
    	#Ap, ASam, accepted = acceptance_prob2(self.pi, pnext, jacp, self.U)
    	self.nnx.zero_grad()
    	self.nnv.zero_grad()
    	#L = loss(Ap, Aq, self.pi, pnext, qi, qnext, self.sc, self.reg)
    	L = loss2(self.train_chain, ASam, 1)
    	#L = loss3(self.pi, pnext)
    	#print("Loss {}".format(L))
    	
    	accepted_sample = ASam.detach().numpy()
    	self.train_chain.append(accepted_sample)
    	self.pi = torch.tensor(accepted_sample).float()
    	L.backward()
    	self.optimizerx.step()
    	self.optimizerv.step()  
    	return L, accepted


    def evolve(self, initial_sample):
    	if self.learning_state==0:
    		self.pi = initial_sample
    		self.learning_state = 1
    		self.chain.append(initial_sample.numpy())
    	pnext, vnext, jacp = self.dyn(self.pi)
    	Ap, ASam, accepted = acceptance_prob(self.pi, pnext, self.vi, vnext, jacp, self.U, self.minf, self.msup)
    	#Ap, ASam, accepted = acceptance_prob2(self.pi, pnext, jacp, self.U)
    	accepted_sample = ASam.detach().numpy()
    	self.chain.append(accepted_sample)    	
    	self.pi = torch.tensor(accepted_sample).float()
    	return accepted


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
	return -_d


def kinetic(_v):
	return torch.sum(0.5*_v**2, dim=1)



def acceptance_prob(state0, state1, vel0, vel1, jac, ufunc, minf=None, msup=None):
	"""calculates acceptance probabilty"""
	_, ndim = state0.shape
	ap = torch.min(torch.zeros(batch_size).float(), - ufunc.value(state1) + ufunc.value(state0) - kinetic(vel1) + kinetic(vel0) + jac)
	
	# 0 probablity for states out of the limits -1e20 for log(0)
	#ap = limits(ap, state1, minf, msup)
	
	# selection of metropolis acceptance rule alpha>u
	un = torch.log(torch.rand((1, batch_size)))

	accepted = torch.t(ap>un).repeat(1, ndim).float()
	naccepted = 1 - accepted

	ap = torch.clamp(torch.exp(ap), 0, 1)
	
	# accepted samples
	acc_sam = state1*accepted + state0*naccepted
	
	return ap, acc_sam, accepted


class Likelihood:
	"""Log likelihood.
  	mod: tensor (batch_size, ndata): model values
  	dat: array (ndata): data values
  	cov: array (ndata, ndata): data error
  	"""
	def __init__(self, _data, _sigma):
		self.n = len(_data)
		self.constant = torch.sum(-0.5*torch.log(2*np.pi*std**2))
		self.data = _data
		self.std = _sigma

	def get_likelihood(self, mod):
		return torch.sum(-0.5*((mod - self.data)/self.std)**2)


class Potential:
  def __init__(self, _data, _cov, _z, _prior):
    """Computes potential energy and its gradient.
    dat: array (ndata), data.
    _cov: array (1, ndata), data error
    z: array (ndata), redshift.
    prior: object, prior.
    """
    self.data = _data
    self.cov = _cov
    self.z = _z
    self.prior = _prior
    self.LikeFunc = Likelihood(_data, _cov)


  def value(self, _theta):
    """Returns potential log value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    _mod = model(_theta, self.z)
    self.u = - self.LikeFunc.get_likelihood(_mod) - self.prior.get_log_pdf(_theta)
    return self.u


  def gradi(self, _theta):
    """Returns gradient value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    theta_grad = torch.tensor(_theta.clone(), requires_grad=True)
    val = self.value(theta_grad)
    gradient, = torch.autograd.grad(val, theta_grad, grad_outputs=torch.ones(_theta.shape[0]), create_graph=True)
    return gradient


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
ndim = 2
ep = args.ep
verbose = args.ver
save = args.save

verboseprint = print if verbose else lambda *a, **k: None

# datos

x = torch.linspace(0, 10, 100)
m = 3
b = 1
mean = torch.Tensor([0])
std = torch.Tensor([1])
datos = m*x + b + torch.randn(len(x))
std = torch.ones(len(x))

# definiciones

mu = torch.Tensor([1., 1.])
cov = torch.eye(2)
prior = Prior(dist="normal", mean=mu, cov=cov)

pot = Potential(datos, std, x, prior)

mlpx = MLP(ndim, neurons, neurons, neurons, 1.0, 1.0)
mlpv = MLP(ndim, neurons, neurons, neurons, 1.0, 1.0)

leapfrog = Leapfrog(U=pot, m=lpsteps, ndim=ndim, nnx=mlpx, nnv=mlpv, e=ep, b=batch_size, reg=reg, sc=scale, lr=lr)

plt.ion()
plt.xlim([m - 1.5, m + 1.5])
plt.ylim([b - 1.5, b + 1.5])
# Train
Losses = []
Accepted = []
for j in range(Nt):
	ti = time.time()
	loss_iter, accepted = leapfrog.train()
	Losses.append(loss_iter.detach().numpy())
	Accepted.append((torch.sum(accepted, dim=0)[0].numpy()))
	tf = time.time()
	verboseprint("Train iteracion {} tiempo {:.2f} Loss {:.2f}".format(j, tf - ti, Losses[j]))#, object(), 3)
	point = leapfrog.train_chain[-1]
	plt.scatter(point[:4, 0], point[:4, 1], color=['navy','green','red', "brown"], marker=".", alpha=0.1)
	plt.pause(1e-4)
Losses = np.array(Losses)
Accepted = np.array(Accepted)
P = np.array(leapfrog.train_chain)
ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if save:
	np.save("samples_train_{}".format(ts), P)
	np.save("loss_train_{}".format(ts), Losses)
	np.save("acceptance_rate_train_{}".format(ts), Accepted)

"""
plt.figure(1)
for j in range(batch_size):
	plt.scatter(P[:, j, 0], P[:, j, 1], marker=".")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("train")

plt.figure(3)
plt.plot(Losses)
plt.title("Loss")
"""

# Evaluation
"""
Accepted = []
initial_p = torch.tensor(pot.prior.get_samples(batch_size)).float()
for j in range(Ne):
	ti = time.time()
	accepted = leapfrog.evolve(initial_p)
	Accepted.append((torch.sum(accepted, dim=0)[0].numpy()))
	tf = time.time()
	verboseprint("Eval iteracion {} tiempo {:.2f}".format(j, tf - ti))
Accepted = np.array(Accepted)
P = np.array(leapfrog.chain)
if save:
	np.save("samples_{}".format(ts), P)
	np.save("acceptance_rate_{}".format(ts), Accepted)
"""
"""
plt.figure(2)
for j in range(batch_size):
	plt.scatter(P[:, j, 0], P[:, j, 1], marker=".")
plt.title("eval")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
"""
"""
plt.ioff()
"""