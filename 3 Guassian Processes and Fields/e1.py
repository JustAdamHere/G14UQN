import math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import pymc3 as pm

def construct_K(k, t, r):
	M = []
	for i in range(len(t)):
		M.append([])
		for j in range(len(t)):
			M[i].append(k(t[i], t[j], r))

	return M

def k1(s, t, r):
	return math.exp(-(s-t)**2/(2*r**2))

def k2(s, t, r):
	return min(s, t)

def k3(s, t, r):
	return sg.unit_impulse(s*t)

def k4(s, t, r):
	return (1 + abs(s-t))*math.exp(-abs(s-t))

n = 100
t = np.linspace(0, 5, n)

k = k4

K_1 = construct_K(k, t, 1)
K_2 = construct_K(k, t, 4)
K_3 = construct_K(k, t, 0.25)

sample_1 = np.random.multivariate_normal(np.zeros(n), K_1)
sample_2 = np.random.multivariate_normal(np.zeros(n), K_2)
sample_3 = np.random.multivariate_normal(np.zeros(n), K_3)

plt.figure(1)
plt.plot(t, sample_1, 'r', label='r=1')
plt.plot(t, sample_2, 'g', label='r=4')
plt.plot(t, sample_3, 'b', label='r=0.25')
plt.legend()
plt.show()