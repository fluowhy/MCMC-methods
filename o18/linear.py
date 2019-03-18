import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def log_like(w, x, y, sd): # - log likelihood
    # aumentar dimension para modelo lineal
    y_est = w[0]*x + w[1]
    return - 0.5*len(x)*np.log(2*np.pi*sd**2) - 0.5*np.sum((y_est - y)**2)/sd**2


def prediccion(y_pred, y):
    """Retorna el total de la muestra, hits y % de precision"""
    total = len(y)
    hits = np.sum(y_pred==y)
    return total, hits, np.around(hits/total, 2)*100


def log_normal(mu, sigma, x):
    dim = len(mu)
    cen = x - mu
    return - 0.5*cen@np.linalg.inv(sigma)@cen.T - np.log((2*np.pi)**dim*np.linalg.det(sigma))
    
    
def met_has(w0, N, X, Y, prior=None, s1=1):  # prior=None->Uniforme
                                                # prior=5->Gaussiana de cov diagonal 5 y mu=w0
    ndim = len(w0)
    sd = 1.5
    p0 = - log_like(w0, X, Y, sd) # "-" para obtener la log likelihood de vuelta
    if prior!=None:
        cov_prior = np.eye(ndim)*prior
        p0 += log_normal(mu=w0, sigma=cov_prior, x=w0)
    chain = []
    i = 0
    j = 0
    while i<N:
        w1 = np.random.multivariate_normal(w0, np.eye(ndim)*s1)
        p1 = - log_like(w1, X, Y, sd)
        if prior!=None:
            p1 += log_normal(mu=w0, sigma=cov_prior, x=w1)
        un = np.log(np.random.uniform(0, 1))
        alpha = min(0, p1 - p0)
        if alpha>=un:
            chain.append(w1)
            p0 = p1
            w0 = w1
            i += 1
        j += 1
    print('acept', i/j)
    return np.array(chain)


dpi = 200

def linear_fun(x, b, m, noise=False, sd=None):
	if noise:
		return m*x + b + np.random.normal(0, sd, len(x))
	else:
		return m*x + b

def random_data(n, b, m, sd):
	x = np.linspace(0, 10, n)
	return x, linear_fun(x, b, m, noise=True, sd=1)

X, Y = random_data(1000, 4, 2, 2)
y = linear_fun(X, 4, 2)

plt.clf()
plt.figure()
plt.scatter(X, Y, marker='.', color='navy', label='datos')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('linear_data', dpi=dpi)

plt.clf()
plt.figure()
plt.scatter(X, Y, marker='.', color='navy', label='datos')
plt.plot(X, y, color='red', label='fit', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = mx + b')
plt.legend()
plt.savefig('linear_fit', dpi=dpi)

x_chain, y_chain = np.random.multivariate_normal(mean=np.array([4, 2]), cov=np.array([[0.3, 0.4], [0.4, 0.6]]), size=10000).T
plt.clf()
plt.figure()
plt.scatter(x_chain, y_chain, color='green', marker='.', alpha='0.5', label=r'mustras $p(\theta|d)$')
plt.scatter(4, 2, label='fit', marker='x', color='black')
plt.legend()
plt.xlabel('b')
plt.ylabel('m')
plt.savefig('state_markov', dpi=dpi)