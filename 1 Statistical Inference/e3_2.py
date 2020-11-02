import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np

def calculate_MLE(data, theta, n):
	MLE = sum(data)/n

	return MLE

def theta_95(data, theta_m):
	n = data.size

	J = sum(data)/(theta_m**2) + (n-sum(data))/(1-theta_m)**2

	interval = [theta_m - 1.96/J, theta_m + 1.96/J]

	return interval

theta = 0.6
contains = []
for n in range(100, 10**4, 1):
	data = np.random.binomial(1, theta, n)
	MLE = calculate_MLE(data, theta, n)
	interval = theta_95(data, MLE)

	if (interval[0] < theta) and (theta < interval[1]):
		contains.append(1)
	else:
		contains.append(0)

print(contains.count(1)/len(contains))