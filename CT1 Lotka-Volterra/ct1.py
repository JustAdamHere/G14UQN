import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

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

# For each alpha, calculates the solution using 
#  a built-in IVP solver on the interval [t0, tf].
#  Uses a RK45 solver.
solution = scipy.integrate.solve_ivp(LotkaVolterra, (t0, tf), initial_condition, args=(10,))

# Separate solution values into t, x, and y.
t = solution.t
x = solution.y[0]
y = solution.y[1]

# QoI for both populations.
x20 = x[-1]
y20 = y[-1]