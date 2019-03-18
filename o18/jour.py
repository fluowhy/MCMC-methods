import numpy as np
import matplotlib.pyplot as plt




theta = np.linspace(-np.pi, np.pi, 200)

p = np.array([1, 2])
plt.scatter(p[0], p[1], color="navy")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh1")

plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.plot(p[0] - np.sin(theta), p[1] - np.cos(theta))
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh2")


p1 = np.random.multivariate_normal(p, np.eye(2), 2)


plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.plot(p[0] - np.sin(theta), p[1] - np.cos(theta))
plt.scatter(p1[0, 0], p1[1, 0], color="red")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh3")


plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.plot(p[0] - np.sin(theta), p[1] - np.cos(theta))
plt.scatter(p1[0, 0], p1[1, 0], color="red")
plt.scatter(p1[0, 0], p1[1, 0], marker="x", color="black")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh4")

plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.plot(p[0] - np.sin(theta), p[1] - np.cos(theta))
plt.scatter(p1[0, 0], p1[1, 0], color="red")
plt.scatter(p1[0, 0], p1[1, 0], marker="x", color="black")
plt.scatter(p1[0, 1], p1[1, 1], color="red")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh5")


P = np.random.multivariate_normal(p, np.array([[1, 0.5], [0.5, 1]]), 1000)
plt.clf()
plt.scatter(P[:, 0], P[:, 1], color="navy", marker=".")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("mh6")


plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("hmc1")

def quad(x, h, k):
	return (x - h)**2 + k

plt.clf()
xs = np.linspace(1.1, 1.5, 5)
px = quad(xs, 1, 2)

plt.clf()
plt.scatter(p[0], p[1], color="navy")
plt.scatter(xs, px, color="green")
plt.scatter(xs[-1], px[-1], color="red")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("hmc2")

plt.clf()
xs = np.linspace(1.1, 1.5, 5)
px = quad(xs, 1, 2)

plt.clf()
p2 = np.random.normal(0, 1)
plt.scatter(p[0], p[1], color="navy")
plt.scatter(p2, p[1], color="red")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("gb1")

plt.clf()
p3 = np.random.normal(0, 1)
plt.scatter(p[0], p[1], color="navy")
plt.scatter(p2, p[1], color="navy")
plt.scatter(p2, p3, color="red")
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.savefig("gb2")

