import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

def pdf(mu, sigma, x):
	return np.real(math.exp(-np.abs(x-mu)/(2*sigma))/(2*sigma))

def likelihood(mu, sigma, x):
	L = []
	for i in range(sigma.size):
		L.append(1)

		for j in range(x.size):
			L[i] *= pdf(mu, sigma[i], x[j])

	return L

sigma = np.linspace(0.1, 15, 100)
mu    = np.linspace(-4, 4, 20)
x     = np.random.laplace(0, 1, 10)

Lm2 = likelihood(-2, sigma, x)
L0  = likelihood(0,  sigma, x)
Lp2 = likelihood(2,  sigma, x)

plt.figure(1)

plt.semilogy(sigma, Lm2)
plt.semilogy(sigma, L0)
plt.semilogy(sigma, Lp2)

plt.show()