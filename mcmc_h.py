# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as sci
from matplotlib.ticker import NullFormatter
from scipy.special import gammainccinv
from scipy.special import erfc
from mpl_toolkits.mplot3d import Axes3D

# Este programa implementa Hamiltonian Markov Chain Monte Carlo. El núcleo utilizado se basa en Metropolis Hastings.

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


def acepta(ec, ep, EC, EP, x, X):
	alpha = min(- EP + ep - ec + EC, 0) # log(alpha)
	u = np.log(np.random.uniform())
	if u<alpha:
		return X, EP
	else:
		return x, ep


def roun(x, d):
	x = x*10**d
	x = np.int(np.rint(x))
	x = x*10**-d
	return x


def EHubble(theta, z): # parametro de hubble
	om0 = theta[0]
	ol = theta[1]
	w = theta[2]
	arg = om0*(1 + z)**3 + (1 - om0 - ol)*(1 + z)**2 + ol*(1 + z)**(3*(1 + w))
	EE = np.sqrt(arg)
	return EE, arg


def modelo(theta,z): # modulo de la distancia teorico
	om0 = theta[0]
	ol = theta[1]
	omega_k = 1 - om0 - ol
	E = EHubble(theta, z)[0]
	I = sci.cumtrapz(1/E, z, initial=0)+z[0]*((1/E)[0] + 1)/2
	o_k_s = np.sqrt(abs(omega_k))
	if omega_k==0:
		dl = (1 + z)*I
	elif omega_k<0:
		dl = (1 + z)*np.sin(o_k_s*I)/o_k_s
	elif omega_k>0:	
		dl = (1 + z)*np.sinh(o_k_s*I)/o_k_s	
	dist = 5*np.log10(dl)
	if (-np.inf==dist).any(): print theta
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
	if bol>0:
		a = 0 # raiz imaginaria 
	else:
		a = 1 # raiz real
	return a


def region(f, x, y, z, alpha, n):
	m = np.int(np.around(alpha*n,0))
	argsort = np.argsort(f)
	xs = x[argsort]
	ys = y[argsort]
	zs = z[argsort]
	xsa = xs[:m]
	ysa = ys[:m]
	zsa = zs[:m]
	return xsa, ysa, zsa
	


def grafica(x, y, xmin, ymin, sx, sy, bins=100): #scatter+marginales
	nullfmt = NullFormatter()         # no labels

	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# start with a rectangular Figure
	plt.figure(1, figsize=(8, 8))

	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)

	# the scatter plot:
	axScatter.scatter(x, y, color='darkred', alpha=.4)
	axScatter.scatter(xmin, ymin, color='black', label='mejor fit '+str(np.around(xmin, 3))+' '+str(np.around(ymin,3)))
	axScatter.legend()

	# now determine nice limits by hand:
	binwidth = 0.25
	xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
	lim = (int(xymax/binwidth) + 1) * binwidth
	a = 0.1
	axScatter.set_xlim((min(x)-a, max(x)+a))
	axScatter.set_ylim((min(y)-a, max(y)+a))
	if sx=='t1':
		sx = '$\Omega_{m,0}$'
	elif sx=='t2':
		sx = '$\Omega_{DE,0}$'
	else:
		sx = 'w'
	if sy=='t1':
		sy = '$\Omega_{m,0}$'
	elif sy=='t2':
		sy = '$\Omega_{DE,0}$'
	else:
		sy = 'w'

	axScatter.set_xlabel(sx)
	axScatter.set_ylabel(sy)

	
	axHistx.hist(x, bins=bins)
	axHisty.hist(y, bins=bins, orientation='horizontal')

	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	plt.show()
	return

# fuente https://matplotlib.org/examples/pylab_examples/scatter_hist.html

def argmin1(t1, t2, t3, z, dat, sig): # calcula el min om, ol, w en base a chi2	
	minchic = np.inf
	a = 0
	for i,j,k in zip(t1,t2,t3):
		mo = modelo(np.array([i,j,k]),z)
		chic = chi2(mo, dat, sig)[0]
		if chic<minchic:
			amin = a
			minchic = chic
		a += 1	
	return t1[amin], t2[amin], t3[amin]


def argmin2(t1, t2, t3, V): # busca en base a vector chi2 y devulve minimos de los parametros	
	amin = np.argmin(V)
	return t1[amin], t2[amin], t3[amin]


def dx2(s, n_p):
	"""
	s: sigmas
	n_p: numero de parametros
	"""
	delta_x_2 = gammainccinv(n_p/2, erfc(s/np.sqrt(2)))
	return delta_x_2


def samples_s(sa1, sa2, x2, d_x2): # determina samples dentro de sigmas
	ind_sam = np.where(x2<=min(x2)+d_x2)
	return sa1[ind_sam], sa2[ind_sam]


def cuenta(x2, W):
	"""
	recibe muestras de una distribucion y X2 para determinar los puntos y X2 para las regiones de credibilidad de 0.68, 0.95 y 0.99 
	"""
	ms = W.shape[1]
	args = np.argsort(x2)
	W = W[:,args]  
	ms68 = int(np.rint(ms*0.68))
	ms95 = int(np.rint(ms*0.95))
	ms99 = int(np.rint(ms*0.99))
	R = np.array([])
	for wi in W:
		R = np.concatenate((R, np.array((wi[:ms68], wi[:ms95], wi[:ms99]))))
	return R, x2[ms68], x2[ms95], x2[ms99]


def potencial(dat, sigma, theta, z):
	u = - likelihood(modelo(theta, z), dat, sigma) - prior(theta) 
	return u


def cinetica(p, m):
	k = np.sum(p**2/2/m)
	return k


def hamiltoniano(p, dat, sigma, theta, z, m=1):
	h = cinetica(p, m) + potencial(dat, sigma, theta, z)
	return h


def gradiente(dw, theta, z, dat, sigma):
	tf = theta + 0.5*dw
	tb = theta - 0.5*dw
	grad = (potencial(dat, sigma, tf, z) - potencial(dat, sigma, tb, z))/dw
	return grad


def leapfrog(pi, l, e, m, dw, theta, z, dat, sigma):
	X = []
	P = []
	X.append(theta)
	P.append(pi)
	qe = theta
	pe = pi
	while True:
		pe = np.random.normal(size=3)
		for i in range(l):		
			pe = pi - 0.5*e*gradiente(dw, qe, z, dat, sigma) # actualiza momento en e/2	
			qe = qe + e*pe/m
			rev = revisa(qe, z)*revisa(qe+0.5*dw, z)*revisa(qe-0.5*dw, z)
			if revisa(qe, z)==0 and revisa(qe+0.5*dw, z)==0 and revisa(qe-0.5*dw, z)==0:
				break
			pe = pe - 0.5*e*gradiente(dw, qe, z, dat, sigma)
			P.append(pe)
			X.append(qe)
			if i+1==l: i+=1
		if i==l: break
	P = np.array(P)
	X = np.array(X)
	return X, P


plt.clf()

"""
redshift = np.genfromtxt('sn1a', usecols=(0)) 
mu_obs = np.genfromtxt('sn1a', usecols=(1)) - M # m - M
cov = np.diag(np.ones(len(redshift))*0.4)
"""
##########################################################################
# Carga de datos
##########################################################################
#names = np.genfromtxt('gal.txt', dtype=str,usecols=(0))
redshift = np.genfromtxt('gal.txt', usecols=(1))
mu_obs = np.genfromtxt('gal.txt', usecols=(2)) # m - M
cov = np.genfromtxt('gal.txt', usecols=(3))

p = np.argsort(redshift)
redshift = redshift[p]
mu_obs = mu_obs[p]
cov = cov[p]
cov = np.diag(cov)

##########################################################################
# Configuracion cadena
##########################################################################
"""____________________________________________________________________"""
M = 1 # numero de cadenas
Chains = []
Xi2 = []
Post = []
params = 3 # numero de parametros
r = 1e-4
m = np.ones(3) # varianza de la energia cinetica, vector de "masas"

print 'parametros', params
print '___________________________________'
"""____________________________________________________"""
for o in range(M):
	print 'cadena ', o+1
	# Matrices de datos de la cadena	
	chain = [] 
	post = [] 
	chi_2 = []
	Ratio = []
	acept = 0
	# Q inicial
	q = np.random.uniform(low=[0,0,-4], high=[2, 2, 0], size=3)	
	mod1 = modelo(q, redshift)
	pos1 = potencial(mu_obs, cov, q, redshift)
	Chi1 = chi2(mod1, mu_obs, cov)[0]
	print q
	chain.append(q)
	post.append(pos1)
	chi_2.append(Chi1)
	Ratio.append(100)	
	
	N = 1000 # numero de muestras

	for i in range(N):
		print i		
		q = chain[i]
		
		# revision si proposal no indefine la busqueda
		while True:
			if i>100:
				p = np.random.normal(loc=np.zeros(3), scale=np.diag(np.cov(np.array(chain).T)), size=3)	
			else: 
				p = np.random.normal(loc=np.zeros(3), scale=m, size=3)	
			Q, P = leapfrog(p, 25, 1e-3, m, 1e-4, q, redshift, mu_obs, cov)
			#H = []			
			#for ip, iq in zip(Q,P):
			#	H.append(hamiltoniano(ip, mu_obs, cov, iq, redshift, m=1))
			#plt.plot(Q[:,0], Q[:,1])
			#plt.scatter(q[0], q[1], color='red')
			#plt.xlim([0, 2])
			#plt.ylim([0, 2])			
			#plt.show()		
			Q1 = Q[-1]
			P1 = P[-1]		
			if revisa(Q1, redshift)==1: # que la raiz no sea imaginaria
				break	
		
		t = cinetica(p, m)
		u = potencial(mu_obs, cov, q, redshift)
		T = cinetica(P1, m)
		U = potencial(mu_obs, cov, Q1, redshift)
		
		A = acepta(t, u, T, U, q, Q1)
		chain.append(A[0])
		post.append(A[1])
		mod1 = modelo(A[0], redshift)
		Chi1 = chi2(mod1, mu_obs, cov)[0]
		chi_2.append(Chi1)	
		# ratio de aceptacion
		acept += tasa(chain[i], chain[i + 1]) 
		Ratio.append(acept/(i+1)*100)
	
	ratio = acept/N*100
	print 'ratio %', ratio

	post = np.array(post)
	chain = np.array(chain)
	chi_2 = np.array(chi_2)
	Ratio = np.array(Ratio)

	t1 = chain[:,0]
	t2 = chain[:,1]
	t3 = chain[:,2]
	


	#t1 = t1[70:] # saca burn in
	#t2 = t2[70:]
	#t3 = t3[70:]

	# buscar argumento del minimo de chi2

	t1m, t2m, t3m = np.around(argmin2(t1, t2, t3, chi_2),3)
	print t1m, t2m, t3m



	#plt.scatter(t1, t2, marker='x', color='black')
	#plt.show()
	#plt.hist(t1, 100)
	#plt.show()
	#plt.hist(t2, 100)
	#plt.show()

	plt.plot(Ratio)
	plt.show()
	"""
	# regiones de confianza


	r1 = dx2(1,2)
	r2 = dx2(2,2)
	r3 = dx2(3,2)

	t1_new_1, t2_new_1 = samples_s(t1, t2, chi_2, r1)
	t1_new_2, t2_new_2 = samples_s(t1, t2, chi_2, r2)
	t1_new_3, t2_new_3 = samples_s(t1, t2, chi_2, r3)

	plt.scatter(t1_new_3, t2_new_3, marker='.', color='navy', label='99%')
	plt.scatter(t1_new_2, t2_new_2, marker='.', color='green', label='95%')
	plt.scatter(t1_new_1, t2_new_1, marker='.', color='red', label='68%')
	plt.scatter(t1m, t2m, marker='x', s=20, color='black', label=('$\Omega_{m,0}$='+str(t1m)+' '+'$\Omega_{\Lambda}$'+str(t2m)))
	plt.title('metodo usando $\chi^{2}$ y gauss')
	plt.xlabel('$\Omega_{m}$')
	plt.ylabel('$\Omega_{\Lambda}$')
	plt.legend()
	plt.show()
	"""


	# regiones metodo 2

	D = np.array((t1, t2, t3))
	vals = cuenta(chi_2, D)
	t1_68 = vals[0][0]
	t1_95 = vals[0][1]
	t1_99 = vals[0][2]
	t2_68 = vals[0][3]
	t2_95 = vals[0][4]
	t2_99 = vals[0][5]
	t3_68 = vals[0][6]
	t3_95 = vals[0][7]
	t3_99 = vals[0][8]

	# grafica de la linea universo plano
	t1min = min(t1)
	t1max = max(t1)

	dom = np.linspace(t1min, t1max, 100)
	rec = 1 - dom

	plt.plot(dom, rec, color='black')
	plt.scatter(t1_99, t2_99, marker='.', color='navy', label='99%')
	plt.scatter(t1_95, t2_95, marker='.', color='green', label='95%')
	plt.scatter(t1_68, t2_68, marker='.', color='red', label='68%')
	plt.title('metodo de conteo')
	plt.xlabel('$\Omega_{m}$')
	plt.ylabel('$\Omega_{\Lambda}$')
	plt.scatter(t1m, t2m, marker='x', s=20, color='black', label=('$\Omega_{m,0}$='+str(t1m)+' '+'$\Omega_{\Lambda}$='+str(t2m)+' '+'$\chi^{2}$='+str(np.around(min(chi_2),3 ))))
	plt.legend()
	plt.show()


	plt.scatter(t1_99, t3_99, marker='.', color='navy', label='99%')
	plt.scatter(t1_95, t3_95, marker='.', color='green', label='95%')
	plt.scatter(t1_68, t3_68, marker='.', color='red', label='68%')
	plt.title('metodo de conteo')
	plt.xlabel('$\Omega_{m}$')
	plt.ylabel('w')
	plt.scatter(t1m, t3m, marker='x', s=20, color='black', label=('$\Omega_{m,0}$='+str(t1m)+' '+'$w=$'+str(t3m)+' '+'$\chi^{2}$='+str(np.around(min(chi_2),3))))
	plt.legend()
	plt.show()


	plt.scatter(t2_99, t3_99, marker='.', color='navy', label='99%')
	plt.scatter(t2_95, t3_95, marker='.', color='green', label='95%')
	plt.scatter(t2_68, t3_68, marker='.', color='red', label='68%')
	plt.title('metodo de conteo')
	plt.xlabel('$\Omega_{\Lambda}$')
	plt.ylabel('w')
	plt.scatter(t2m, t3m, marker='x', s=20, color='black', label=('$\Omega_{\Lambda}$='+str(t2m)+' '+'$w=$'+str(t3m)+' '+'$\chi^{2}$='+str(np.around(min(chi_2),3))))
	plt.legend()
	plt.show()
		
	Super = np.c_[chain[:,0],chain[:,1],chain[:,2],chi_2,post,Ratio]	
	
	Chains.append(chain)
	Post.append(post)
	Xi2.append(chi_2)
	
	
Chains = np.array(Chains)
Post = np.array(Post)
Xi2 =  np.array(Xi2)

# covarianza entre parametros de distintas cadenas
"""
for i in range(params):
	print i
	plt.clf()
	covp = np.cov(Chains[:,:,i])
	plt.imshow(covp, origin='lower')
	plt.colorbar()
	plt.xlabel('cadena')
	plt.ylabel('cadena')
	if i==0:
		par = '$\Omega_{m}$'
		part = 'om'
	elif i==1:
		par = '$\Omega_{\Lambda}$'
		part = 'ol'
	elif i==2:
		par = '$w$'
		part = 'w'
	plt.title('parametro '+par)
	plt.savefig('cov_'+part)
"""
"""
grafica(t1, t2, t1m, t2m, 't1', 't2') # scatter y marginales
grafica(t1, t3, t1m, t3m, 't1', 't3') 
grafica(t2, t3, t2m, t3m, 't2', 't3')
"""


"""
myrange1 = [min(t1), max(t1), min(t2), max(t2)]
myrange2 = [min(t1), max(t1), min(t3), max(t3)]
myrange3 = [min(t2), max(t2), min(t3), max(t3)]
# curvas de nivel
bins = 100
H,[x,y,z] = np.histogramdd((t1, t2, t3), bins=(bins, bins, bins), range=[[min(t1), max(t1)], [min(t2), max(t2)], [min(t3), max(t3)]]) 

dx = abs(max(x)-min(x))
dy = abs(max(y)-min(y))
dz = abs(max(z)-min(z))
cte = np.sum(H)*dx*dy*dz
H = H/cte
Hxy = np.sum(H, axis=2)*dz
Hxz = np.sum(H, axis=0)*dy
Hyz = np.sum(H, axis=1)*dx
plt.imshow(Hxy.T, origin='low',extent=myrange1,interpolation='nearest', aspect='auto')
plt.show()
xn=(x[1:]+x[:-1])/2
yn=(y[1:]+y[:-1])/2
zn=(z[1:]+z[:-1])/2
amin = np.argmin(chi_2)
t1min = np.argmin(abs(t1[amin]-xn))
t2min = np.argmin(abs(t2[amin]-yn))
t3min = np.argmin(abs(t3[amin]-zn))

hs = np.linspace(np.min(H), np.max(H), 2000)

min99 = np.inf
min95 = np.inf
min68 = np.inf
for i in hs:
	alpha = np.sum((H>=i)*H)*dx*dy*dz
	dif99 = abs(alpha-0.99)
	dif95 = abs(alpha-0.95)
	dif68 = abs(alpha-0.68)
	if dif99<=min99:	
		min99 = dif99
		h99 = i
		a99 = alpha
	if dif95<=min95:	
		min95 = dif95
		h95 = i
		a95 = alpha
	if dif68<=min68:	
		min68 = dif68
		h68 = i
		a68 = alpha
print 'a99', a99
print 'a95', a95
print 'a68', a68

levels = (h99, h95, h68)

# om, ol
plt.clf()
plt.scatter(t1, t2, marker='.', color='black')
plt.contour(Hxy.T, levels, origin='lower', colors=['red','green','blue'], extent=myrange1, linewidth=4)
plt.legend()
plt.scatter(t1m, t2m, marker='x', s=20, color='brown', label=('$\Omega_{m,0}$='+str(t1m)+' '+'$\Omega_{DE,0}$='+str(t2m)))
plt.xlabel('$\Omega_{m,0}$')
plt.ylabel('$\Omega_{DE,0}$')
plt.legend()
plt.savefig('r_om_ol')

# ol, w
plt.clf()
plt.scatter(t2, t3, marker='.', color='black')
plt.contour(Hxz.T, levels, origin='lower', colors=['red','green','blue'], extent=myrange3, linewidth=4)
plt.scatter(t2m, t3m, marker='x', s=20, color='brown', label=('$\Omega_{DE,0}$='+str(t2m)+' '+'$w=$'+str(t3m)))
plt.ylabel('$w$')
plt.xlabel('$\Omega_{DE,0}$')
plt.legend()
plt.savefig('r_ol_w')

# om, w
plt.clf()
plt.scatter(t1, t3, marker='.', color='black')
plt.contour(Hyz.T, levels, origin='lower', colors=['red','green','blue'], extent=myrange2, linewidth=4)
plt.scatter(t1m, t3m, marker='x', s=20, color='brown', label=('$\Omega_{m,0}$='+str(t1m)+' '+'$w=$'+str(t3m)))
plt.xlabel('$\Omega_{m,0}$')
plt.ylabel('$w$')
plt.legend()
plt.savefig('r_om_w')
"""

#-------------------------------------------------------------------
#___________________________________________________________________


