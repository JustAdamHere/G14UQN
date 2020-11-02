import math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

def construct_K(k, t, v, r, alpha):
	M = []
	for i in range(len(t)):
		N = M.append([])
		for j in range(len(t)):
			M[i].append(k(t[i], t[j], v, r, alpha))

	return M

def k(s, t, v, r, alpha):
	return v*math.exp(-(s-t)**alpha/r**alpha)

n = 100
t = np.linspace(0, 5, n)

black_K = construct_K(k, t, 1,   2,   2)
blue_K  = construct_K(k, t, 1,   2,   0.8)
red_K   = construct_K(k, t, 1.8, 3,   1.2)
green_K = construct_K(k, t, 1.4, 0.5, 2)

black_sample = np.random.multivariate_normal(np.zeros(n), black_K)
#blue_sample  = np.random.multivariate_normal(np.zeros(n), blue_K)
#red_sample   = np.random.multivariate_normal(np.zeros(n), red_K)
green_sample = np.random.multivariate_normal(np.zeros(n), green_K)

plt.figure(1)
plt.plot(t, black_sample, 'k')
plt.plot(t, green_sample, 'g')
plt.show()