import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

# Probability density function.
def pdf(mu, sigma, x):
	term1 = 1/math.sqrt(2*math.pi*sigma**2)
	term2 = -1/(2*sigma**2)*(x - mu)**2

	return term1*math.exp(term2)

# Likelihood function.
def likelihood(mu, data):
	sigma = 1

	L = []
	for i in range(mu.size):
		L.append(1)

		for j in range(data.size):
			L[i] *= pdf(mu[i], 1, data[j])

	return L

###############
## Plot of L ##
###############

# Range of mu and data.
mu = np.linspace(-1, 1, 13)
data = np.random.normal(0, 1, 100)

# Plots on semilog axis.
plt.figure(1)
#plt.semilogy(mu, likelihood(mu, data), label="Test")

######################
## MLE Calculations ##
######################

data20 = np.random.normal(0.5, 1, 20)
data40 = np.random.normal(0.5, 1, 40)
data100 = np.random.normal(0.5, 1, 100)
data200 = np.random.normal(0.5, 1, 200)

plt.plot(mu, np.log(likelihood(mu, data20)),  'r-', label="n=20")
plt.plot(mu, np.log(likelihood(mu, data40)),  'g-', label="n=40")
plt.plot(mu, np.log(likelihood(mu, data100)), 'b-', label="n=100")
plt.plot(mu, np.log(likelihood(mu, data200)), 'k-', label="n=200")

plt.plot(mu[np.argmax(likelihood(mu, data20))], np.log(max(likelihood(mu, data20))),  'rx')
plt.plot(mu[np.argmax(likelihood(mu, data40))], np.log(max(likelihood(mu, data40))),  'rx')
plt.plot(mu[np.argmax(likelihood(mu, data100))], np.log(max(likelihood(mu, data100))), 'rx')
plt.plot(mu[np.argmax(likelihood(mu, data200))], np.log(max(likelihood(mu, data200))), 'rx')

plt.grid(True)
plt.xlabel("mu")
plt.ylabel("L")
plt.title("Exercise 1 (1)")
plt.legend(loc="lower right")

plt.show()