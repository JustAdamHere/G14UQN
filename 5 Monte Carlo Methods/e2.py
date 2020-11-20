import math
import matplotlib.pyplot as plt
import numpy as np

###################
## MC ESTIMATION ##
###################
def MCEstimator(t, u0, lam):
	value = 0

	for i in range(len(lam)):
		value += exact(t, u0, lam[i])

	return value/len(lam)

def MCExact(t, u0, lam_dag, epsilon):
	return -u0*(math.exp(-(lam_dag + epsilon)*t) - math.exp(-(lam_dag - epsilon)*t))/(2*epsilon*t)

def varianceEstimator(t, u0, lam, MCEstimate):
	value = 0

	for i in range(len(lam)):
		value += (exact(t, u0, lam[i]) - MCEstimate)**2

	return value/(len(lam) - 1)

def varianceExact(t, u0, lam_dag, epsilon):
	return -u0**2*(math.exp(-2*(lam_dag + epsilon)*t) - math.exp(-2*(lam_dag - epsilon)*t))/(4*epsilon*t) - u0**2*(math.exp(-(lam_dag + epsilon)*t) - math.exp(-(lam_dag - epsilon)*t))**2/(4*epsilon**2*t**2)

def probabilityExact(t, u0, lam_dag, epsilon, u):
	if (u0*math.exp(-t*(lam_dag + epsilon)) <= u and u<= u0*math.exp(-t*(lam_dag - epsilon))):
		return 1/(2*epsilon*t*u)
	else:
		return 0

def exact(t, u0, lam):
	return u0*math.exp(-lam*t)

###################
## GP REGRESSION ##
###################
def compute_K(k, s, t):
	M = np.zeros((len(s), len(t)))
	for i in range(len(s)):
		for j in range(len(t)):
			M[i][j] = k(s[i], t[j])

	return M

def compute_posteriorMean(k, data, sigma, point):
	x = data[:, 0]
	y = data[:, 1]

	K1 = compute_K(k, [point], x)
	K2 = compute_K(k, x,       x)
	I = np.identity(len(K2))

	middleTerm = np.linalg.inv(K2 + sigma**2*I)
	return K1.dot(middleTerm).dot(y)[0]

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

##################
## ACTUAL STUFF ##
##################



#################
## GP FIT PLOT ##
#################
B = 7
t = 5

######################
## GP SAMPLE VS PDF ##
######################