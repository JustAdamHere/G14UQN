import numpy as np
import scipy.integrate



def LotkaVolterra(z, t, alpha, beta, gamma, delta):
	x = z[0]
	y = z[1]

	result = np.zeros(2)

	result[0] = alpha*x - beta*x*y
	result[1] = delta*x*y - gamma*y

	return result

def solve(y0, t_range, t, ):
	return 0

# Problem data.
t_min = 0
t_max = 10
t_range = np.linspace(t_min, t_max, 100)

###############################
## WOULDN'T ORDINARILY EXIST ##
###############################
# Parameters for simulated data.
parameters = (1, 1, 1, 1)
x0 = 1
y0 = 1

# Noise.
a = np.array([13, 15])

# Simulated solution.
simulated_data = scipy.integrate.odeint(LotkaVolterra, [x0, y0], t_range, args=parameters)

data = np.zeros((simulated_data.shape[0], 3))
for i in range(data[:, 0].size):
	data[i][0] = simulated_data[i][0] + np.random.normal(0, a[0])
	data[i][1] = simulated_data[i][1] + np.random.normal(0, a[1])
	data[i][2] = t_range[i]
###############################
###############################
###############################