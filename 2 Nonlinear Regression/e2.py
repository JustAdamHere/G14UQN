import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def new_rate(conc, beta1, beta2):
	return beta1 + beta2/conc

def Michaelis_Menten(conc, theta1, theta2):
	return theta1*conc/(theta2 + conc)

def Michaelis_Menten_1(conc, theta1, theta2):
	grad1 = conc/(theta2 + conc)
	grad2 = -theta1*conc/(theta2 + conc)**2
	return [grad1, grad2]

def Michaelis_Menten_2(conc, theta1, theta2):
	grad11 = 0
	grad12 = -conc/(theta2 + conc)**2
	grad21 = grad12
	grad22 = theta1*conc/(theta2 + conc)**3
	return [[grad11, grad12], [grad21, grad22]]

## Input data.
concentration = np.array([2.856829, 5.005303, 7.519473, 22.101664, 27.769976, 39.198025, 45.483269, 203.784238])
rate          = np.array([14.58342, 24.74123, 31.34551, 72.96985,  77.50099,  96.08794,  96.96624,  108.88374])

## Initial guess of parameters.
beta, _ = scipy.optimize.curve_fit(new_rate, concentration, 1/rate, [1, 1])
guess = [1/beta[0], beta[1]/beta[0]]

## Parameter calculation.
theta, _ = scipy.optimize.curve_fit(Michaelis_Menten, concentration, rate, guess)

## Estimated data.
estimated_data = []
for i in range(concentration.size):
	estimated_data.append(Michaelis_Menten(concentration[i], theta[0], theta[1]))

## UQ.
theta_1 = Michaelis_Menten_1(0.8, theta[0], theta[1])
theta_2 = Michaelis_Menten_2(0.8, theta[0], theta[1])

J_inv = np.linalg.inv(theta_2)
print([theta - J_inv.dot([1.96, 1.96]), theta + J_inv.dot([1.96, 1.96])]) # I do not think that this is the correct interval calculation.
# Instead need to use the function given in the notes.

## Plotting.
plt.figure(1)
plt.title('True rate and estimated rate against concentration')
plt.plot(concentration, rate,           label='True rate')
plt.plot(concentration, estimated_data, label='Estimated rate')
plt.legend()
plt.show()