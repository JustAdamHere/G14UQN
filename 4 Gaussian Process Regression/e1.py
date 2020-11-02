import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def compute_K(k, s, t):
	M = np.zeros((len(s), len(t)))
	for i in range(len(s)):
		for j in range(len(t)):
			M[i][j] = k(s[i], t[j])

	return M

def full_k(s, t, r, alpha):
	return math.exp(-((s-t)/r)**alpha)

def compute_posteriorMean(k, data, sigma, point):
	x = data[:, 0]
	y = data[:, 1]

	K1 = compute_K(k, [point], x)
	K2 = compute_K(k, x,       x)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(y)[0]

def compute_posteriorMu(k, data, new_x, sigma):
	x = data[:, 0]
	y = data[:, 1]

	K1 = compute_K(k, new_x, x)
	K2 = compute_K(k, x,     x)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(y)

def compute_posteriorSigma(k, data, new_x, sigma):
	x = data[:, 0]
	y = data[:, 1]

	K1 = compute_K(k, new_x, new_x)
	K2 = compute_K(k, new_x, x)
	K3 = compute_K(k, x,     x)
	K4 = compute_K(k, x,     new_x)
	I = np.identity(len(K3))

	middleTerm = np.linalg.inv(K3 + sigma**2*I)
	return K1 - K2.dot(middleTerm).dot(K4)

def compute_posteriork(k, data, sigma):
	def new_k(s, t):
		x = data[:, 0]
		y = data[:, 1]

		K1 = compute_K(k, [s], x)
		K2 = compute_K(k, x,   x)
		K3 = compute_K(k, x,   [s])
		I = np.identity(len(K2))

		middleTerm = np.linalg.inv(K2 + sigma**2*I)
		return k(s, t) - K1.dot(middleTerm).dot(K3)

	return new_k

############
## PART 1 ##
############
# Existing data.
data = np.array([[0.9, 0.1], [3.8, 1.2], [5.2, 2.1], [6.1, 1.1], [7.5, 1.5], [9.6, 1.2]])
sigma = math.sqrt(0.15)

# Grid spacing and covariance. 
new_x = np.linspace(0, 10, 50)

# Our standard k.
k = lambda s, t: full_k(s, t, math.sqrt(2), 2)

# Posterior mean at each point.
m = np.zeros(len(new_x))
for i in range(len(new_x)):
	m[i] = compute_posteriorMean(k, data, sigma, new_x[i])

# Posterior k, and confidence band calculation.
posteriork = compute_posteriork(k, data, sigma)
confidence95 = np.zeros((len(new_x), 2))
for i in range(len(new_x)):
	confidence95[i][0] = m[i] - 1.96*posteriork(new_x[i], new_x[i])
	confidence95[i][1] = m[i] + 1.96*posteriork(new_x[i], new_x[i])

# Posterior mu and sigma.
posteriorMean     = compute_posteriorMu   (k, data, new_x, sigma)
posteriorVariance = compute_posteriorSigma(k, data, new_x, sigma)
new_y = np.random.multivariate_normal(posteriorMean, posteriorVariance) 

# Plots data, f estimate, and confidence bands.
plt.figure(1)
plt.plot(data[:, 0], data[:, 1], 'rx',           label='Data')
plt.plot(new_x, new_y,           color='orange', label='Estimate f')
plt.fill_between(
	new_x,confidence95[:, 0], confidence95[:, 1],
	                             color='gray'
)

############
## PART 2 ##  <-- this could all be wrong...
############
# Grid points.
new_x = np.sort(np.random.uniform(10, 11.5, 10))

# Posterior mean at each point.
m = np.zeros(len(new_x))
for i in range(len(new_x)):
	m[i] = compute_posteriorMean(k, data, sigma, new_x[i])

# Posterior k, and confidence band calculation.
posteriork = compute_posteriork(k, data, sigma)
confidence95 = np.zeros((len(new_x), 2))
for i in range(len(new_x)):
	confidence95[i][0] = m[i] - 1.96*posteriork(new_x[i], new_x[i])
	confidence95[i][1] = m[i] + 1.96*posteriork(new_x[i], new_x[i])

# Posterior mu and sigma.
posteriorMean     = compute_posteriorMu   (k, data, new_x, sigma)
posteriorVariance = compute_posteriorSigma(k, data, new_x, sigma)
new_y = np.random.multivariate_normal(posteriorMean, posteriorVariance) 

plt.figure(2)
plt.plot(data[:, 0], data[:, 1], 'rx',           label='Data')
plt.plot(new_x, new_y,           color='orange', label='Estimate f')
plt.fill_between(
	new_x,confidence95[:, 0], confidence95[:, 1],
	                             color='gray'
)

############
## PART 3 ##
############
# Choices of parameters.
r     = np.array([0.1, 0.5, 1, 2, 5])
alpha = np.array([1, 2])

# Grid spacing and covariance. 
new_x = np.linspace(0, 10, 50)

# Our standard k.
k = lambda s, t: full_k(s, t, math.sqrt(2), 2)

# Posterior mean at each point.
m = np.zeros(len(new_x))
for i in range(len(new_x)):
	m[i] = compute_posteriorMean(k, data, sigma, new_x[i])

# Posterior mu and sigma.
new_y = np.zeros((len(r), len(alpha), len(new_x)))
for i in range(len(r)):
	for j in range(len(alpha)):
		posteriorMean     = compute_posteriorMu   (k, data, new_x, sigma)
		posteriorVariance = compute_posteriorSigma(k, data, new_x, sigma)

		new_y[i][j][:] = np.random.multivariate_normal(posteriorMean, posteriorVariance)

# Plots data, f estimate, and confidence bands.
plt.figure(3)
for i in range(len(r)):
	for j in range(len(alpha)):
		colormap = matplotlib.cm.get_cmap('autumn')
		rgba = colormap(r[i]/max(r))
		rgba = rgba[:3] + (alpha[j]/max(alpha),)
		plt.plot(new_x, new_y[i][j], color=rgba, label='r = '+str(r[i])+', a = '+str(alpha[j]))
plt.plot(data[:, 0], data[:, 1], 'rx', label='Data')
plt.legend()

plt.show()
#plt.savefig("e1")