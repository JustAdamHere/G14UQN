import math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np

def construct_K(k, t):
	M = []
	for i in range(len(t)):
		M.append([])
		for j in range(len(t)):
			M[i].append(k(t[i], t[j]))
	return M

def k(x, y):
	return min(x, y)

def basis(i, x):
	j = i+1
	return math.sqrt(2)*math.sin(j - math.pi/(2*x))
	#return math.sqrt(2)*math.sin((j - 1/2)*math.pi/x)
	#return math.sqrt(2)*math.sin((j-math.pi)/(2*x))

def eigenvalue(i):
	j = i+1
	#return 0.25/((2*math.pi*j - math.pi)**2)
	return 1/((j-1/2)*math.pi)**2
	#return 0.25*(2*math.pi*j - math.pi)**(-2)

def construct_function(c, M, x):
	value = 0
	for i in range(M):
		value += c[i]*basis(i, x)

	return value

########################
### KL DECOMPOSITION ###
########################
## Grid Points.
n = 100
chi = np.linspace(0.01, 1, n)

## Random coefficients.
M = 1000
c = []
for i in range(M):
	c.append(np.random.normal(0, eigenvalue(i)))

## Random GP functions.
f = []
for i in range(len(chi)):
	f.append(construct_function(c, M, chi[i]))

##############################
### STANDARD GP GENERATION ###
##############################
K = construct_K(k, chi)

sample = np.random.multivariate_normal(np.zeros(n), K)

plt.figure(1)
plt.plot(chi, f)
plt.plot(chi, sample)
plt.show()