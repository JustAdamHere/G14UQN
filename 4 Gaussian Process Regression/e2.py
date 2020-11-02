import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def compute_K(k, s, t):
	M = np.zeros((len(s), len(t)))
	for i in range(len(s)):
		for j in range(len(t)):
			M[i][j] = k(s[i], t[j])

	return M

def full_k(s, t, theta):
	nu     = theta[0]
	r      = theta[1]
	sigma2 = theta[2]

	value = nu*math.exp(-((s-t)/r)**2)

	if s==t:
		value += sigma2 # For kroncker delta.

	return value

def compute_posteriorMean(k, data, sigma, point):
	x = data[:, 0]
	y = data[:, 1]

	K1 = compute_K(k, [point], x)
	K2 = compute_K(k, x,       x)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(y)[0]

def logp_maximiser(full_k, data):
	x  = data[:, 0]
	y  = data[:, 1]
	yT = np.transpose(y)

	n = len(x)

	def maximiser(theta):
		K    = compute_K(lambda s, t: full_k(s, t, theta), x, x)
		Kinv = np.linalg.inv(K)
		Kdet = np.linalg.det(K)

		return -(yT.dot(Kinv).dot(y) + math.log(abs(Kdet)) + n*math.log(2*math.pi))/2

	return maximiser

############
## PART 1 ##
############
# Data and parameters.
data = np.array([[0.9, 0.1], [3.8, 1.2], [5.2, 2.1], [6.1, 1.1], [7.5, 1.5], [9.6, 1.2]])
theta = (1, 2, 0.03)

# X axis range.
x_range = np.linspace(0, 10, 100)

# Our k.
k = lambda s, t: full_k(s, t, theta)

# Posterior mean function.
posteriorMean = lambda x: compute_posteriorMean(k, data, theta[2], x)

# Confidence bands.
confidence95 = np.zeros((len(x_range), 2))
for i in range(len(x_range)):
	x = x_range[i]
	confidence95[i][0] = posteriorMean(x) - 1.96*k(x, x)
	confidence95[i][1] = posteriorMean(x) + 1.96*k(x, x)

# Plots the posterior mean.
fig, axis = plt.subplots(2)
for x in x_range:
	axis[0].plot(x, posteriorMean(x), 'bo')
axis[0].plot(data[:, 0], data[:, 1], 'rx')
axis[0].fill_between(x_range, confidence95[:, 0], confidence95[:, 1], color='gray')

############
## PART 2 ##
############
# Maximiser/minimiser.
maximiser = logp_maximiser(full_k, data)
minimiser = lambda theta: -maximiser(theta)

# Maximal parameters.
optimisationResult = scipy.optimize.minimize(minimiser, x0=theta, method='Nelder-Mead')
print(optimisationResult)
theta_max = optimisationResult.x
print('\nEstimate of θ: ' + str(theta_max))

# Updated k.
k = lambda s, t: full_k(s, t, theta_max)

# Updated posterior mean function.
posteriorMean = lambda x: compute_posteriorMean(k, data, theta_max[2], x)

# Confidence bands.
confidence95 = np.zeros((len(x_range), 2))
for i in range(len(x_range)):
	x = x_range[i]
	confidence95[i][0] = posteriorMean(x) - 1.96*k(x, x)
	confidence95[i][1] = posteriorMean(x) + 1.96*k(x, x)

# Plots the updated posterior mean.
for x in x_range:
	axis[1].plot(x, posteriorMean(x), 'bo')
axis[1].plot(data[:, 0], data[:, 1], 'rx')
axis[1].fill_between(x_range, confidence95[:, 0], confidence95[:, 1], color='gray')

plt.show()