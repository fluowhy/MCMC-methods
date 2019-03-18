# -*- coding: utf-8 -*-
# https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
# bash ~/Anaconda3-5.2.0-Linux-x86_64.sh

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as sci
import time
import os
from getdist import plots, MCSamples
import datetime

import tensorflow as tf
ds = tf.contrib.distributions

global batch_size
global dpi
dpi = 200

tf.reset_default_graph()

def f1(theta, z, omk):
    zc = np.copy(z)
    zc = np.insert(zc, 0, 0)
    dz = zc[1:] - zc[:-1]
    E = EHubble(theta, z)[0]
    I = tf.cumsum(dz/(E + 1e-300), axis=1)
    o_k_s = tf.reshape(tf.sqrt(abs(omk)), [batch_size, 1])
    return (1 + z)*tf.sinh(o_k_s*I)/(o_k_s + 1e-300)


def f2(theta, z, omk):
    zc = np.copy(z)
    zc = np.insert(zc, 0, 0)
    dz = zc[1:] - zc[:-1]
    E = EHubble(theta, z)[0]
    I = tf.cumsum(dz/(E + 1e-300), axis=1)
    o_k_s = tf.reshape(tf.sqrt(abs(omk)), [batch_size, 1])
    return (1 + z)*tf.sin(o_k_s*I)/(o_k_s + 1e-300)


def f3(theta, z, omk):
    zc = np.copy(z)
    zc = np.insert(zc, 0, 0)
    dz = zc[1:] - zc[:-1]
    E = EHubble(theta, z)[0]
    I = tf.cumsum(dz/(E + 1e-300), axis=1)
    return (1 + z)*I


def EHubble(theta, z): # parametro de hubble
    """
    theta: parameter space state.
    z: redshift.
    bs: batch size.
    """
    bs = batch_size
    om0 = theta[:, 0]
    ol = theta[:, 1]
    w = theta[:, 2]
    ts = tf.shape(theta)
    zz = np.tile(z, (bs, 1))
    arg = tf.reshape(om0, [ts[0], 1])*(1 + z)**3 + tf.reshape((1 - om0 - ol), [ts[0], 1])*(1 + z)**2 + tf.reshape(ol, [ts[0], 1])*(1 + z)**(3*(1 + tf.reshape(w, [ts[0], 1])))
    EE = tf.sqrt(arg)
    return EE, arg


def modelo(theta, z):    
    om0 = theta[:, 0]
    ol = theta[:, 1]
    w = theta[:, 2]    
    omega_k = 1 - om0 - ol
    sig = tf.sign(omega_k)
    may = tf.reshape(1 + tf.sign(sig - 1), [batch_size, 1])
    men = tf.reshape(1 - tf.abs(sig), [batch_size, 1])
    eq = tf.reshape(1 - tf.sign(sig + 1), [batch_size, 1])    
    dl = may*f1(theta, z, omega_k) + eq*f3(theta, z, omega_k) + men*f2(theta, z, omega_k)
    # integral
    dist = 5*tf.log(dl + 1e-300)/np.log(10)
    return dist


def likelihood(mod, dat, sigma):
  """Log likelihood.
  mod: tensor (batch_size, ndata): model values
  dat: array (ndata): data values
  cov: array (ndata, ndata): data error
  """
  L = - 0.5*chi2(mod, dat, sigma)[0] + np.sum(-0.5*np.log(np.diag(2*np.pi*sigma**2)))
  return L


def chi2(mod, dat, cov):
  """
  mod: tensor (batch_size, ndata): model values
  dat: array (ndata): data values
  cov: array (ndata, ndata): data error
  """
  cov_p = np.diag(cov)
  AA = tf.reduce_sum(((dat - mod)/cov_p)**2, axis=1)
  BB = tf.reduce_sum((dat - mod)/cov_p**2, axis=1)
  CC = np.sum(1/cov_p**2)
  chi = AA - (BB**2)/CC
  return chi, BB/CC

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
    """Returns potential value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    mod = modelo(theta, self.z)
    self.u = - likelihood(mod, self.data, self.cov) - self.prior.get_log_pdf(theta) 
    return self.u

  def grad(self, theta):
    """Returns gradient value in a point or batch of points.
    theta: tensor (batch_size, ndim), point in parameter space.
    """
    self.value(theta)
    self.gradient = tf.gradients(self.u, theta)
    return self.gradient[0]

class Prior:
  def __init__(self, dist=None, low=None, high=None, mean=None, cov=None):
    """Defines some priors. Uniform and normal.
    dist: str, distribution name.
    low: array or list (ndim), uniform low limit.
    high: array or list (ndim), uniform high limit.
    mean: array or list (ndim), normal dist. mean.
    cov: array or list (ndim, ndim), normal dist. covariance.
    """
    self.dist = dist 
    if dist==None:
        self.u = tf.distributions.Uniform(low=low, high=high)
        vol = np.prod(np.abs((high - low)))
        self.pdf = tf.ones(batch_size)/vol
        self.logpdf = tf.log(self.pdf)
    elif dist=='normal':
        self.u = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    
  def get_samples(self, n):
    """Get distribution samples.
    n: int, # of samples, batch size.
    """
    return tf.cast(self.u.sample(sample_shape=(n)), tf.float32)

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


class L2HMC:
    def __init__(self, U, prior, b, m, lr, sc, reg, lfp, lfq, x0):
        """
        U: energy function
        prior: prior distribution
        b: batch size
        m: leapfrog steps
        lr: learning rate
        sc: scale parameter
        reg: regularization parameter
        lfp: inital samples leapfrog object
        lfq: batch samples leapfrog object
        x0: initial position in parameter space
        """
        self.U = U
        self.prior = prior
        self.b = b
        self.m = m
        self.lr = lr
        self.sc = sc
        self.reg = reg
        self.init_samples = self.prior.get_samples(b)
        self.lfp = lfp
        self.lfq = lfq
        self.X = x0 # ep
        self.optimizer = tf.train.AdamOptimizer(lr)
    
        
    def distance(self):
        print('distance')
        self.dp = tf.reduce_sum((self.X - self.X1)**2, axis=1)
        self.dq = tf.reduce_sum((self.Xq - self.xq)**2, axis=1)
    
    def acceptance(self):
        print('acceptance')

        
    def loss(self, d): 
        return self.sc**2/d - d/self.sc**2
        
    def Loss(self):
        print('loss')
        self.distance()
        self.acceptance()
        self.L0ss = (tf.reduce_mean(self.loss(self.dp)) 
                     + tf.reduce_mean(self.reg*self.loss(self.dq)))
        
    def train(self):
        self.Loss()
        print('opti')
        self.optimizer.minimize(self.L0ss)

class MLP:
    def __init__(self, ndim, n1, n2, ls, lq):
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
        self.ls = ls
        self.lq = lq
    
    def model(self, x, v, t):
        self.h1 = (tf.contrib.layers.fully_connected(inputs=x, num_outputs=self.n1, activation_fn=None) + 
              tf.contrib.layers.fully_connected(inputs=v, num_outputs=self.n1, activation_fn=None) + 
              tf.contrib.layers.fully_connected(inputs=t, num_outputs=self.n1, activation_fn=None))
        self.h1 = tf.nn.relu(self.h1)
        
        self.h2 = tf.contrib.layers.fully_connected(inputs=self.h1, num_outputs=self.n2, activation_fn=tf.nn.relu)
        
        self.S = self.ls*tf.contrib.layers.fully_connected(inputs=self.h2, num_outputs=self.ndim, activation_fn=tf.nn.tanh)
        self.Q = self.lq*tf.contrib.layers.fully_connected(inputs=self.h2, num_outputs=self.ndim, activation_fn=tf.nn.tanh)
        self.T = tf.contrib.layers.fully_connected(inputs=self.h2, num_outputs=self.ndim, activation_fn=None)

def mlp(x, v, t, n1, n2, ndim):
  h1 = (tf.contrib.layers.fully_connected(inputs=x, num_outputs=n1, activation_fn=None) + 
              tf.contrib.layers.fully_connected(inputs=v, num_outputs=n1, activation_fn=None) + 
              tf.contrib.layers.fully_connected(inputs=t, num_outputs=n1, activation_fn=None))
  h1 = tf.nn.relu(h1)

  h2 = tf.contrib.layers.fully_connected(inputs=h1, num_outputs=n1, activation_fn=tf.nn.relu)
  
  h2 = tf.contrib.layers.fully_connected(inputs=h2, num_outputs=n2, activation_fn=tf.nn.relu)

  S = tf.contrib.layers.fully_connected(inputs=h2, num_outputs=ndim, activation_fn=tf.nn.tanh)
  Q = tf.contrib.layers.fully_connected(inputs=h2, num_outputs=ndim, activation_fn=tf.nn.tanh)
  T = tf.contrib.layers.fully_connected(inputs=h2, num_outputs=ndim, activation_fn=None)
  
  epsi = 0.01*tf.contrib.layers.fully_connected(inputs=h2, num_outputs=ndim, activation_fn=tf.sigmoid)
  return S, Q, T, epsi

"""#### CODE"""

class Leapfrog:
    """Leapfrog object, solve dynamics."""
    def __init__(self, U, m, ndim, nnx, nnv, e, b):
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
        #self.x = tf.Variable(tf.random_normal([b, ndim]))
        #self.v = tf.Variable(tf.random_normal([b, ndim]))
        #self.t = tf.Variable(tf.random_normal([b, 2]))
        self.d = np.random.choice([-1, 1])
        
    def direction(self):
        """Samples a new direction"""
        self.d = np.random.choice([-1, 1])
        
        
    def for_dyn_fun(self):
        """One step of forward dynamics d=1.
        """
        # remember S, Q, T update in each sub iteration
        self.grad = self.U.grad(self.x)
        self.Sv, self.Qv, self.Tv, self.e = mlp(self.x, self.grad, self.t, 50, 50, 3) 
                
        self.v = tf.squeeze(self.v*tf.exp(0.5*self.Sv*self.e) - 0.5*self.e*(self.grad*tf.exp(self.Qv*self.e) + self.Tv))
        
        self.Sx, self.Qx, self.Tx, self.e = mlp(self.x, self.v, self.t, 50, 50, 3) 
                
        self.x = tf.squeeze(tf.multiply(self.x, self.nmask) + tf.multiply(self.mask, self.x*tf.exp(self.e*self.Sx) + 
            self.e*(self.v*tf.exp(self.e*self.Qx) + self.Tx)))
        
        self.Sx, self.Qx, self.Tx, self.e = mlp(self.x, self.v, self.t, 50, 50, 3) 
        
        self.x = tf.squeeze(tf.multiply(self.x, self.mask) + 
                            tf.multiply(self.nmask, self.x*tf.exp(self.e*self.Sx) +
                                        self.e*(self.v*tf.exp(self.e*self.Qx) + self.Tx)))
        
        self.grad = self.U.grad(self.x)
        self.Sv, self.Qv, self.Tv, self.e = mlp(self.x, self.grad, self.t, 50, 50, 3)
        
        self.v = tf.squeeze(self.v*tf.exp(0.5*self.Sv*self.e) - 0.5*self.e*(self.grad*tf.exp(self.Qv*self.e) + self.Tv))
        
        
    def back_dyn_fun(self):
        """One step of backward dynamics d=-1.
        """
        self.grad = self.U.grad(self.x)
        self.Sv, self.Qv, self.Tv, self.e = mlp(self.x, self.grad, self.t, 50, 50, 3) 
        
        self.v = tf.squeeze(self.v*tf.exp(- 0.5*self.Sv*self.e) + 0.5*self.e*(self.grad*tf.exp(self.Qv*self.e) + self.Tv))
        
        self.Sx, self.Qx, self.Tx, self.e = mlp(self.x, self.v, self.t, 50, 50, 3) 

        self.x = tf.squeeze(tf.multiply(self.x, self.mask) + tf.multiply(self.nmask, self.x*tf.exp(- self.e*self.Sx) - 
            self.e*(self.v*tf.exp(self.e*self.Qx) + self.Tx)))
        
        self.Sx, self.Qx, self.Tx, self.e = mlp(self.x, self.v, self.t, 50, 50, 3) 
        
        self.x = tf.squeeze(tf.multiply(self.x, self.nmask) + tf.multiply(self.mask, self.x*tf.exp(- self.e*self.Sx) - 
            self.e*(self.v*tf.exp(self.e*self.Qx) + self.Tx)))
        
        self.grad = self.U.grad(self.x)
        
        self.Sv, self.Qv, self.Tv, self.e = mlp(self.x, self.grad, self.t, 50, 50, 3)
        
        self.v = tf.squeeze(self.v*tf.exp(- 0.5*self.Sv*self.e) + 0.5*self.e*(self.grad*tf.exp(self.Qv*self.e) + self.Tv))
        
        
    def dyn(self, x):
        """m steps dynamics"""
        self.resample()
        self.x = x
        for i in range(self.m):
            self.mask = np.random.choice([1., 0.], size=(self.b, self.ndim), p=[0.5, 0.5])
            self.nmask = np.ones((self.b, self.ndim))*1. - self.mask
            self.mask = tf.convert_to_tensor(self.mask, dtype=tf.float32)
            self.nmask = tf.convert_to_tensor(self.nmask, dtype=tf.float32)
            # update outputs of the neural network
            self.time_encoding(i) # updates time
            if self.d==1:
                self.for_dyn_fun()
            elif self.d==-1:
                self.back_dyn_fun()
        self.flip()
        return self.x
     
    def get_value(self):
        return self.x
          
                
    def time_encoding(self, mi):
        """Encodes time.
        mi: int, actual leapfrog step.
        """
        arg = 2*np.pi*mi/self.m
        val = np.array([np.cos(arg), np.sin(arg)])
        val = np.tile(val, (self.b, 1))
        val = tf.convert_to_tensor(val)
        self.t = tf.cast(val, tf.float32)
        
        
    def resample(self):
        """Resamples velocity and direction"""
        self.v = tf.random_normal(shape=(self.b, self.ndim), mean=0.0, stddev=1.0, 
                                  dtype=tf.float32)
        self.direction()
        
        
    def flip(self):
        """Flip direction"""
        self.d *= -1


class Loss:
  def __init__(self, reg, sc, lr, u):
    """
    sc: float, scale parameter.
    reg: float, regularization parameter.
    lr: float, learning rate.
    u: object, potential energy function.
    """
    self.reg = reg
    self.sc = sc
    self.optimizer = tf.train.AdamOptimizer(lr)
    self.u = u   
  

  def acceptance(self):
    self.ap = tf.minimum(0., - self.u.value(self.ep2) + self.u.value(self.ep1))
    self.aq = tf.minimum(0., - self.u.value(self.eq2) + self.u.value(self.eq1))
    
    # 0 probablity for states out of the limits
    self.ap = limits(self.ep2, self.ap)
    self.aq = limits(self.eq2, self.aq)
    
    un = tf.log(tf.random_uniform(shape=(batch_size, 1), dtype=tf.float32))
    self.act = tf.maximum(0., tf.sign(un - tf.reshape(self.ap, (batch_size, 1))))
    self.nact = tf.ones_like(self.act) - self.act
    
    self.ap = tf.exp(self.ap) + 1e-20
    self.aq = tf.exp(self.aq) + 1e-20
       
    self.acc = self.nact*self.ep2 + self.act*self.ep1


  def lam(self, d, a):
    #a = 1.0
    return self.sc**2/d/a - d*a/self.sc**2
       
    
  def evaluate(self):
    self.distance()
    self.acceptance()
    self.loss = tf.reduce_mean(self.lam(self.dp, self.ap)) + tf.reduce_mean(self.reg*self.lam(self.dq, self.aq))
  
  def train(self, ep1, ep2, eq1, eq2):
    self.ep1 = ep1
    self.ep2 = ep2
    self.eq1 = eq1
    self.eq2 = eq2
    self.evaluate()
    self.optimizer.minimize(self.loss)
    return self.loss, self.acc, tf.squeeze(self.nact)
    
  def distance(self):
    self.dp = tf.reduce_sum((self.ep1 - self.ep2)**2, axis=1)
    self.dq = tf.reduce_sum((self.eq1 - self.eq2)**2, axis=1)

"""

def l(lamb, distance, acc):
  res = lamb**2/distance/acc
  return res - 1/res


def distance(x1, x2):
  return tf.reduce_sum((x1 - x2)**2, axis=1)


 
def acceptance(x1, x2, u):
    ax = tf.minimum(0., - u.value(x2) + u.value(x1))
    
    
    # 0 probablity for states out of the limits
    ax = limits(x2, ax)
    
    
    un = tf.log(tf.random_uniform(shape=(batch_size, 1), dtype=tf.float32))
    act = tf.maximum(0., tf.sign(un - tf.reshape(ax, (batch_size, 1))))
    nact = tf.ones_like(act) - act
    
    ax = tf.exp(ax) + 1e-20
       
    acc = nact*x2 + act*x1
    return acc, ax, tf.squeeze(nact)


def loss(l1, l2, lamb_b):
  return tf.reduce_mean(l1) + lamb_b*tf.reduce_mean(l2)


def Loss(p, p1, q, q1, u, lamb, lamb_b):
  # Calculates loss
  dp = distance(p, p1)
  dq = distance(q, q1)
  acp, pap, nactp = acceptance(p, p1, U) #LOSS.train(p, p1, q, q1) #
  acq, paq, nactq = acceptance(q, q1, U)
  lp = l(lamb, dp, pap)
  lq = l(lamb, dq, paq)
  return loss(lp, lq, lamb_b), acp, nactp
"""

def limits(vec, vec2mask):
    mask1 = tf.cast(vec[:, 0]<=1.0, tf.float32)*tf.cast(vec[:, 0]>=0.0, tf.float32)
    mask2 = tf.cast(vec[:, 1]<=1.0, tf.float32)*tf.cast(vec[:, 1]>=0.0, tf.float32)
    mask3 = tf.cast(vec[:, 2]<=-1/3, tf.float32)*tf.cast(vec[:, 2]>=-4.0, tf.float32)
    #mask = tf.transpose(tf.stack([mask1, mask2, mask3]))
    mask = mask1*mask2*mask3 # tf.reduce_prod(mask, axis=1)
    nmask = tf.ones_like(mask) - mask
    #return vec2mask*mask + tf.log(mask)
    return vec2mask*mask - nmask*tf.ones_like(nmask)*1e20


def acceptance(p1, p2, u):
    ap = tf.minimum(0., - u.value(p2) + u.value(p1))
    
    ap = limits(p2, ap)
    
    un = tf.log(tf.random_uniform(shape=(batch_size, 1), dtype=tf.float32))
    act = tf.maximum(0., tf.sign(un - tf.reshape(ap, (batch_size, 1))))
    nact = tf.ones_like(act) - act
    
    ap = tf.exp(ap)
    
    acc = nact*p2 + act*p1
    
    return acc, tf.squeeze(nact)
  
  
def ratio(A):
    B = np.array(A)
    a1, _ = B.shape
    return np.sum(B, axis=0)/a1
  
  
def save(array, name):
    from google.colab import files
    # save training chains
    np.save(name, array)
    files.download('{}.npy'.format(name)) 
    return
  
  
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

# carga de datos

direc = "/home/mromero/gal.txt"
#direc = "/home/mauricio/Documents/Uni/otoño_2018/Intro_2/gal.txt"

redshift = np.genfromtxt(direc, usecols=(1))
mu_obs = np.genfromtxt(direc, usecols=(2)) # m - M
cov = np.genfromtxt(direc, usecols=(3))

p = np.argsort(redshift)
redshift = redshift[p].astype(np.float32)
mu_obs = mu_obs[p]
cov = cov[p]
cov = np.diag(cov)


tf.reset_default_graph()
"""
# train parameters from paper
N = 5000
lr = 1e-3
batch_size = 200
"""
N = 40000 # chain steps
lr = 1e-3 # learning rate
batch_size = 40
nneurons = 2
sc = 1.0
reg = 1.0
ls = 1.0
lq = 1.0
lfstep = 1e-2
lfsteps = 8#10
optimizer = tf.train.AdamOptimizer(lr)

"""
text_file = open("hyperparameters.txt", "w")
text_file.write("chain_steps: {}\nlearning_rate: {}\nbatch_size: {}\nneurons: {}\nsc: {}\nreg: {}\nls: {}\nlq: {}\nlfstep: {}\nlfsteps: {}".format(N, lr, batch_size, nneurons, sc, reg, ls, lq, lfstep, lfsteps))
text_file.close()
os.remove("hyperparameters.txt")
"""

# some definitions
low = np.array([0., 0., - 4.])
high = np.array([1., 1., - 1/3])
prior_dist = Prior(dist=None, low=low, high=high)
U = Potential(dat=mu_obs, cov=cov, z=redshift, prior=prior_dist)
Nnx = MLP(ndim=3, n1=nneurons, n2=nneurons, ls=ls, lq=lq)
Nnv = MLP(ndim=3, n1=nneurons, n2=nneurons, ls=ls, lq=lq)
lpp = Leapfrog(U=U, m=lfsteps, ndim=3, nnx=Nnx, nnv=Nnv, e=lfstep, b=batch_size)
lpq = Leapfrog(U=U, m=lfsteps, ndim=3, nnx=Nnx, nnv=Nnv, e=lfstep, b=batch_size)
LOSS = Loss(sc=sc, reg=reg, lr=lr, u=U)

p = tf.placeholder(tf.float32, shape=(None, 3))
p_new = np.random.uniform(low=low, high=high, size=(batch_size, 3))

q = prior_dist.get_samples(batch_size)
p_next = lpp.dyn(p)
q_next = lpq.dyn(q)

train = LOSS.train(p, p_next, q, q_next)

# tensorflow stuff
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)  

# list of states
points = []
losses = []
accept = []
points.append(p_new)
accept.append(np.ones(batch_size))

ti = time.time()
for i in range(N):
  loss_act, p_new, acce = sess.run(train, {p: p_new}) 
  points.append(p_new)
  accept.append(acce)
  losses.append(loss_act)
  print('Chain step: {} Loss = {} Accept. ratio = {:.2f}'.format(i, loss_act, np.median(ratio(accept))))
  #print("Points {}".format(p_new))
tfi = time.time()
print('Training time {:1f} min.'.format((tfi - ti)/60))

"""
plt.clf()
plt.figure(figsize=(18, 10))
print(len(losses))
plt.step(np.arange(len(losses)), np.log(losses), color='navy')
plt.xlabel('epoch')
plt.ylabel('log loss')
plt.savefig('Loss')
"""
ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
points = np.array(points)
np.save("chain_points_train_{}".format(ts), points)

"""
names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]
maxs = np.max(np.max(points, axis=0), axis=0)
mins = np.min(np.min(points, axis=0), axis=0)
for i in range(batch_size):
  plot(points[:, i, 0], points[:, i, 1], points[:, i, 2], names, save=True, savename='l2hmc_train_{}.png'.format(i))
"""


N = 50000 # chain steps

p = tf.placeholder(tf.float32, shape=(None, 3))
p_new = np.random.uniform(low=low, high=high, size=(batch_size, 3))
p_next = lpp.dyn(p)
accp = acceptance(p, p_next, U)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)  
Points = []
Points.append(p_new)
Accept = []
Accept.append(np.ones(batch_size))

ti = time.time()
for i in range(N):
  #p_new, _, acce = sess.run(accp, {p: p_new})
  p_new, acce = sess.run(accp, {p: p_new})
  Points.append(p_new)
  Accept.append(acce)
  print('Chain step: {} Accept. ratio = {:.2f}'.format(i, np.median(ratio(Accept))))
tfi = time.time()
print('Evaluation time {:1f} min.'.format((tfi - ti)/60))

Points = np.array(Points)
np.save("chain_points_{}".format(ts), Points)


"""
names = [r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$\omega$"]
maxs = np.max(np.max(Points, axis=0), axis=0)
mins = np.min(np.min(Points, axis=0), axis=0)
for i in range(batch_size):
  plot(Points[:, i, 0], Points[:, i, 1], Points[:, i, 2], names, save=True, savename='l2hmc_{}.png'.format(i))
"""