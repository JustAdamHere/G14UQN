import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

# Probability density function.
def pdf(alpha, beta, x):
	term1 = 1/(math.gamma(alpha)*beta**alpha)
	term2 = x**(alpha-1)
	term3 = math.exp(-x/beta)

	if x > 0:
		return term1*term2*term3
	else:
		return 0

# Likelihood function.
def likelihood(alpha, beta, data):
	sigma = 1

	L = []
	for i in range(alpha.size):
		L.append([])

		for j in range(beta.size):
			L[i].append(1)

			for k in range(data.size):
				L[i][j] *= pdf(alpha[i], beta[j], data[k])

	return L

###############
## Plot of L ##
###############

# Range of parameters.
alpha = np.linspace(1.1, 25, 50)
beta  = np.linspace(1.1, 10, 50)
A, B  = np.meshgrid(alpha, beta)

# Data.
data10  = np.random.gamma(2, 1, 10)
data50  = np.random.gamma(2, 1, 50)
data100 = np.random.gamma(2, 1, 100)

L10  = likelihood(np.log(alpha), np.log(beta), data10)
L50  = likelihood(np.log(alpha), np.log(beta), data50)
L100 = likelihood(np.log(alpha), np.log(beta), data100)

# Plots on semilog axis.
fig = plt.figure(1)
axis = fig.add_subplot(2, 2, 1, projection='3d')
axis.plot_surface(A, B, np.transpose(L10),  cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
axis = fig.add_subplot(2, 2, 2, projection='3d')
axis.plot_surface(A, B, np.transpose(L50),  cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
axis = fig.add_subplot(2, 2, 3, projection='3d')
axis.plot_surface(A, B, np.transpose(L100), cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
axis.set_xlabel('alpha')
axis.set_ylabel('beta')
axis.set_zlabel('likelihood')

plt.show()

# Do surface plot: https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html