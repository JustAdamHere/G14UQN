import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

def calculate_MLE(theta, n):
	data = np.random.binomial(1, theta, n)

	MLE = sum(data)/n

	return MLE

theta = 0.6
E = []
n_range = np.linspace(1, 5000, 200).astype(np.int)
for n in n_range:
	MLE = calculate_MLE(theta, n)
	E.append(np.abs(MLE-theta))

plt.figure(1)
plt.plot(n_range, E)
plt.show()