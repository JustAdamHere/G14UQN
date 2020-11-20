import math
import matplotlib.pyplot as plt
import numpy as np

def w(t, u0, lam_o, lam_e):
	return (exact(t, u0, lam_o) + exact(t, u0, lam_e))/2

def varianceExact(t, u0, lam_dag, epsilon):
	return -u0**2*(math.exp(-2*(lam_dag + epsilon)*t) - math.exp(-2*(lam_dag - epsilon)*t))/(4*epsilon*t) - u0**2*(math.exp(-(lam_dag + epsilon)*t) - math.exp(-(lam_dag - epsilon)*t))**2/(4*epsilon**2*t**2)

def exact(t, u0, lam):
	return u0*math.exp(-lam*t)

############
## PART 1 ##
############
# Number of samples.
J = 100

# Sample of lambdas.
lam_dag = 0.5
epsilon = 0.4
lam_o = np.random.uniform(lam_dag - epsilon, lam_dag + epsilon, J)
lam_e = 2*lam_dag - lam_o

# Covariance calculation.
covariance = np.cov(lam_o, lam_e)
print(covariance)

############
## PART 2 ##
############
# Problem parameters.
t  = 5
u0 = 1

w_sample = np.zeros(J)
for i in range(len(w_sample)):
	w_sample[i] = w(t, u0, lam_o[i], lam_e[i])

w_estimate = sum(w_sample)/J