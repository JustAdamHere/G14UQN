import math
import matplotlib.pyplot as plt
import numpy.random
import scipy.integrate
import scipy.interpolate
import scipy.optimize

def logistic(y, t, r, c):
	return r*y*(1-y/c)

def true_sol(t, r, c):
	a = (c-2)/2
	return c/(1+a*math.exp(-r*(t-1)))

def wow(y, r, c):
	return logistic(y, 0, r, c)

def solve(y0, t_range, t, r, c):
	solution = scipy.integrate.odeint(logistic, y0, t_range, args=(r, c)).squeeze()
	linear_interpolant = scipy.interpolate.interp1d(t_range, solution)
	return linear_interpolant(t)

t0 = 1
tN = 10
y0 = 2

t_range = numpy.linspace(t0, tN, 100)

###############################
## WOULDN'T ORDINARILY EXIST ##
###############################
r0 = 2
c0 = 100

simulated_data = scipy.integrate.odeint(logistic, y0, t_range, args=(r0, c0))

y = []
for i in range(simulated_data.size):
	y.append(simulated_data[i][0] + numpy.random.normal(0, 13))
###############################
###############################
###############################

fitted_parameters, fitted_covariance = scipy.optimize.curve_fit(lambda t, r, c: solve(y0, t_range, t, r, c), t_range, y)

r_estimate = fitted_parameters[0]
c_estimate = fitted_parameters[1]

print(fitted_parameters)

y_estimate = scipy.integrate.odeint(logistic, y0, t_range, args=(r_estimate, c_estimate))

plt.figure(1)
plt.plot(t_range, simulated_data, 'b', label='Simulated data')
plt.plot(t_range, y,              'k', label='Noisy data')
plt.plot(t_range, y_estimate,     'r', label='Estimated data')
plt.legend()
plt.show()