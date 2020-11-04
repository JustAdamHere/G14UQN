import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d

# LOTKAVOLTERRA
# Gives the derivative of the populations at a given
#  current population.
#
# Inputs:
# 	t 		Current time (irrelevant here).
#   point	Current (x, y) population.
#   alpha	Prey growth rate.
#
# Outputs:
#	x_		dx/dt at (x, y) with alpha.
# 	y_		dy/dt at (x, y) with alpha.
#
def LotkaVolterra(t, point, alpha):
	x = point[0]
	y = point[1]

	x_ = alpha*x - 0.5*x*y
	y_ = 0.6*x*y - y

	return x_, y_

# k variance.
def k(x, y):
	return math.exp(-0.5*(x-y)**2)

# Computes K matrix.
def compute_K(k, s, t):
	M = np.zeros((len(s), len(t)))
	for i in range(len(s)):
		for j in range(len(t)):
			M[i][j] = k(s[i], t[j])

	return M

# Computes posterior mean.
def compute_posteriorMean(k, alpha, QoI, new_alpha, sigma):
	K1 = compute_K(k, [new_alpha], alpha)
	K2 = compute_K(k, alpha,       alpha)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(QoI)[0]

# Computes posterior k.
def compute_posteriork(k, alpha, sigma):
	def new_k(s, t):
		K1 = compute_K(k, [s],   alpha)
		K2 = compute_K(k, alpha, alpha)
		K3 = compute_K(k, alpha,   [s])
		I = np.identity(len(K2))

		middleTerm = np.linalg.inv(K2 + sigma**2*I)
		return k(s, t) - K1.dot(middleTerm).dot(K3)

	return new_k

# Computes posterior mu.
def compute_posteriorMu(k, alpha, QoI, new_alpha, sigma):
	K1 = compute_K(k, new_alpha, alpha)
	K2 = compute_K(k, alpha,         alpha)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(QoI)

# Computes posterior sigma.
def compute_posteriorSigma(k, alpha, new_alpha, sigma):
	K1 = compute_K(k, new_alpha, new_alpha)
	K2 = compute_K(k, new_alpha, alpha)
	K3 = compute_K(k, alpha,     alpha)
	K4 = compute_K(k, alpha,     new_alpha)
	I = np.identity(len(K3))

	middleTerm = np.linalg.inv(K3 + sigma**2*I)
	return K1 - K2.dot(middleTerm).dot(K4)

###############
## MAIN CODE ##
###############
# Number of samples for alpha.
n = 10

# Samples alpha linearly in [5, 10].
alpha = np.linspace(5, 10, n)

# Time range in [t0, tf].
t0 = 0
tf = 20

# Initial conditions for prey and predator populations.
x0 = 5
y0 = 8
initial_condition = (x0, y0)

# Storage for quantities of interest.
QoI = np.zeros(n)    # Quantities of interest.

# For each alpha, calculates the solution using 
#  a built-in IVP solver on the interval [t0, tf].
for i in range(n):
	#  Uses a RK45 solver to solve.
	solution = scipy.integrate.solve_ivp(LotkaVolterra, (t0, tf), initial_condition, args=(alpha[i],))

	# Separate solution values into t, x, and y.
	t = solution.t
	x = solution.y[0]
	y = solution.y[1]

	# QoI, x(20).
	x20 = x[-1]
	QoI[i] = x20

# Storage for various items, and variance of the error.
sigma        = math.sqrt(0.15)     # Variance.
m            = np.zeros(n)    # Posterior mean (and estimation of g).
confidence99 = np.zeros((n, 2)) # Upper and lower 99% bands.

pk = compute_posteriork(k, alpha, sigma)

# Loops for each alpha.
for i in range(n):
	# Posterior k and posterior mean.
	
	pm   = compute_posteriorMean(k, alpha, QoI, sigma, alpha[i])

	# Stores the posterior mean as estimate of g.
	m[i] = pm

	# 99% confidence interval for this alpha.
	#  2.576 is the z-score given for a 99% confidence.
	confidence99[i][0] = pm - 2.576*pk(alpha[i], alpha[i])
	confidence99[i][1] = pm + 2.576*pk(alpha[i], alpha[i])

# Fits a GP to fit the data around alpha=7.5.
posteriorMean = compute_posteriorMu   (k, alpha, QoI, alpha, sigma)
posteriorVari = compute_posteriorSigma(k, alpha,      alpha, sigma)
g = np.random.multivariate_normal(posteriorMean, posteriorVari)
g_range = np.random.multivariate_normal(posteriorMean, posteriorVari, 100)

# Estimate for all range of gs generated.
g75 = np.zeros(100)
for i in range(100):
	g75[i] = interp1d(alpha, g)(7.5)

# Plots estimate of g.
plt.figure(1)
plt.plot(alpha, QoI, 'rx', label='True x(20)')
plt.plot(alpha, confidence99[:, 0], 'r--')
plt.plot(alpha, confidence99[:, 1], 'r--')
plt.plot(alpha, g, 'b-', label='g')
plt.xlabel('alpha')
plt.ylabel('x(20)')
plt.title('Plot of estimate, g, with 99 percent confidence bands')
plt.legend()

# Histogram data.
g_estimates = []
for i in range(100): # 100 new g samples.
	g75 = interp1d(alpha, g)(7.5)
	g_estimates.append(g75)

# Plots histogram of g(7.5)
plt.figure(2)
hist = np.hstack((g75, g75))
plt.hist(hist)
plt.xlabel('g')
plt.ylabel('frequency')
plt.title('Histogram of posterior estimate g(7.5)')

plt.show()

## COMMENTS ##
# Alpha affects the growth rate of the prey, so by changing this value we see different resultant populations for the prey at t=20.
#  Although the confidence bands in my code do not work, we notice that the estimate, g, does fit the value of x(20) pretty closely in
#  that region (unlike near alpha=6.5). I think therefore that there is a good chance that g near alpha=7.5 will give a reasonable
#  approximation to the true value at t=20. 
#
# My histogram doesn't work, but I was hoping to see a higher frequency in that plot for values that peak in the middle. This would
#  show that changing alpha doesn't drastically change the QoI value.