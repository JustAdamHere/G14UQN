import math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np

def alpha_range(k, t, r, n, alpha_min, alpha_max, m):
	h = (alpha_max - alpha_min)/(n-1)

	alpha = []
	for i in range(m):
		alpha.append(alpha_min + i*h)

	K = []
	for i in range(m):
		K.append(construct_K(k, t, r, alpha[i]))

	sample = []
	for i in range(m):
		sample.append(np.random.multivariate_normal(np.zeros(n), K[i]))

	return K, sample, alpha

def plot_range(plt, t, sample_range, color, alpha):
	for i in range(len(sample_range)):
		plt.plot(t, sample_range[i], color + '--', alpha=alpha[i])

def construct_K(k, t, r, alpha):
	M = []
	for i in range(len(t)):
		M.append([])
		for j in range(len(t)):
			M[i].append(k(t[i], t[j], r))

	return M

def k(s, t, r):
	return math.exp(-(s-t)**2/(2*r**2))

n = 100
t = np.linspace(0, 5, n)

## Standard calculation.
K_1 = construct_K(k, t, 1,    1)
K_2 = construct_K(k, t, 4,    1)
K_3 = construct_K(k, t, 0.25, 1)

sample_1 = np.random.multivariate_normal(np.zeros(n), K_1)
sample_2 = np.random.multivariate_normal(np.zeros(n), K_2)
sample_3 = np.random.multivariate_normal(np.zeros(n), K_3)

## Range of alpha.
K_1_range, sample_1_range, alpha_1 = alpha_range(k, t, 1,    n, 0.1, 2, 3)
K_2_range, sample_2_range, alpha_2 = alpha_range(k, t, 4,    n, 0.1, 2, 3)
K_3_range, sample_3_range, alpha_3 = alpha_range(k, t, 0.25, n, 0.1, 2, 3)

## Plot both types.
plt.figure(1)
plt.plot(t, sample_1, 'r', label='r=1')
plt.plot(t, sample_2, 'g', label='r=4')
plt.plot(t, sample_3, 'b', label='r=0.25')
plot_range(plt, t, sample_1_range, 'r', alpha_1)
plot_range(plt, t, sample_2_range, 'g', alpha_2)
plot_range(plt, t, sample_3_range, 'b', alpha_3)
plt.legend()
plt.show()