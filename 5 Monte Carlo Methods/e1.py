import math
import matplotlib.pyplot as plt
import numpy as np

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

############
## Part 1 ##
############
# Number of samples.
J = 100

# Sample of lambdas.
lam_dag = 0.5
epsilon = 0.4
lam_range = np.random.uniform(lam_dag - epsilon, lam_dag + epsilon, J)

# Problem parameters.
t  = 5
u0 = 1

# Estimators.
mc_estimator  = MCEstimator(t, u0, lam_range)
var_estimator = varianceEstimator(t, u0, lam_range, mc_estimator)

# Exacts.
mc_exact  = MCExact(t, u0, lam_dag, epsilon)
var_exact = varianceExact(t, u0, lam_dag, epsilon)

# Prints esimates.
print("MC estimator:  " + str(mc_estimator))
print("MC exact:      " + str(mc_exact))
print("Var estimator: " + str(var_estimator))
print("Var exact:     " + str(var_exact))

############
## PART 2 ##
############
# u and p range for the plot of the exact probability.
u_range = np.linspace(0, 1, 100)
p_range = np.zeros(100)
for i in range(len(p_range)):
	p_range[i] = probabilityExact(t, u0, lam_dag, epsilon, u_range[i])

# Range of choice of Js.
J_range = np.rint(np.logspace(1, 5, num=100)).astype(int)

# Sample lambdas.
lam_range = np.random.uniform(lam_dag - epsilon, lam_dag + epsilon, J_range[-1])

# Get the corresponding us for these lambdas.
u_sample = np.zeros(len(lam_range))
for i in range(len(u_sample)):
	u_sample[i] = exact(t, u0, lam_range[i])

# Plot histogram of sampled us, as well as the exact probability distribution.
plt.figure(1)
plt.hist(u_sample, bins=20, density=True)
plt.plot(u_range, p_range)

############
## PART 3 ##
############
# Range of Js to use.
J_range = np.rint(np.logspace(1, 6, num=200)).astype(int)

LHS = np.zeros(len(J_range))
RHS = np.zeros(len(J_range))
for i in range(len(LHS)):
	lam_range = np.random.uniform(lam_dag - epsilon, lam_dag + epsilon, J_range[i])
	RHS[i] = varianceExact(t, u0, lam_dag, epsilon)/J_range[i]
	LHS[i] = (MCEstimator(t, u0, lam_range) - MCExact(t, u0, lam_dag, epsilon))**2

# Plots log values. Question asks for loglog values but I think this seems to kind of work.
plt.figure(2)
plt.plot(J_range, np.log(RHS), label='RHS')
plt.plot(J_range, np.log(LHS), label='LHS')
plt.legend()
plt.xlabel('J')
plt.show()