# linear_regression.py

'''
This file contains an implementation of logistic regression. 
Code based on the project provided by the Coursera, Machine Learning course by Andrew Ng.
Written in Python rather than Octave or Matlab.
'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import random
import os
import argparse

def generateData(num_features, num_samples, X_range):
	"""
	:type num_features: int
	:type num_samples: int
	:type X_range: (int, int)
	:rtype: array_like, array_like

	returns random generated data
	X => random training data
	y => random output data

	*** Will likely result in bad predictions because data is random!
	"""

	def getRandomPoint(r):
		return (float(random.randint((r[0] + 1) * 1000, (r[1] + 1) * 1000)) / 1000.0) - 1.0
	
	X = np.array([[getRandomPoint(X_range) for i in range(num_features)] for j in range(num_samples)])
	y = np.array([random.randint(0,1) for i in range(num_samples)])
	return X,y

def sigmoid(z):
	"""
	:type z: float OR array_like
	:rtype: float OR array_like

	if input is a scalar, function returns the sigmoid value
	if input is a tensor, return tensor of sigmoid values for each index
	"""
	# convert input to a numpy array
	z = np.array(z)

	# You need to return the following variables correctly 
	g = np.zeros(z.shape)

	if z.ndim == 0:
		return (1.0 / (1.0 + np.exp(-z)))
	elif z.ndim == 1:
		g = np.array([(1.0 / (1.0 + np.exp(-z[i]))) for i in range(z)])
	else:
		g = np.array([[(1.0 / (1.0 + np.exp(-z[i][j]))) for j in range(z.shape[1])] for i in range(z.shape[0])])

	return g

def costFunction(theta, X,  y):
	"""
	:type theta: array_like
	:type X: array_like
	:type y: array_like
	:rtype: float, array_like

	returns cost after optimization, and the gradient
	"""
	m = y.size
	J = 0
	grad = np.zeros(theta.shape)

	for i in range(m):
		H_x = sigmoid(theta.T.dot(X[i]))
		J += (y[i] * np.log(H_x)) + ((1.0 - y[i]) * np.log(1.0 - H_x))
	J *= (-1.0 / m)

	for j in range(theta.shape[0]):
		for i in range(m):
			H_x = sigmoid(theta.T.dot(X[i]))
			grad[j] = (H_x - y[i]) * X[i][j]

		grad[j] /= m

	return J, grad

def costFunctionReg(theta, X, y, lambda_=1.0):
	"""
	:type theta: array_like
	:type X: array_like
	:type y: array_like
	:rtype: float, array_like

	returns cost after optimization, and the gradient using regularization
	"""
	# Initialize some useful values
	m = y.size  # number of training examples

	# You need to return the following variables correctly 
	J = 0
	grad = np.zeros(theta.shape)

	# Calculate cost
	for i in range(m):
		H_x = sigmoid(theta.T.dot(X[i]))
		J += (y[i] * np.log(H_x)) + ((1.0 - y[i]) * np.log(1.0 - H_x))
	J *= (-1.0 / m)

	# Calculate gradient
	theta_squared_sum = 0
	for j in range(theta.shape[0]):
		for i in range(m):
			H_x = sigmoid(theta.T.dot(X[i]))
			grad[j] += (H_x - y[i]) * X[i][j]

		grad[j] /= m

		if j > 0:
			theta_squared_sum += theta[j] ** 2
			# Add regularization parameter
			grad[j] += (lambda_ / float(m)) * theta[j]

	J += (lambda_ / (2.0*m)) * theta_squared_sum
	return J, grad

def predict(theta, X):
	"""
	:type test_data: array_like
	:type theta: array_like
	:rtype: array_like

	Uses calculated values of theta to make a prediction given new sample points.
	If the training data was normalized, make sure test data is too!
	"""
	
	m = X.shape[0]
	p = np.zeros(m)
	for i in range(m):
		p[i] = 1 if sigmoid(theta.T.dot(X[i])) >= 0.5 else 0

	return p

def main():
	# Provide user the option of showing plots
	ap = argparse.ArgumentParser()
	ap.add_argument('--plot', action='store_true', help="Show plots of data when available.")
	args = ap.parse_args()

	# n => number of features, m => sample size
	n,m = 2, 30

	# This data is random; to get a good prediction, use real sample data
	X,y = generateData(n, m, (0,20))
	
	pos = y == 1
	neg = y == 0

	# Plot the training data
	if args.plot:
		plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
		plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
		plt.title("Training data")
		plt.show()

	# Add intercept term to X
	X = np.concatenate([np.ones((m, 1)), X], axis=1)
	initial_theta = np.zeros(n+1)

	# Use the SciPy module to mimize theta with respect to the cost function (rather than iterative gradient descent)
	res1 = optimize.minimize(costFunction,
		initial_theta,
		(X,y),
		jac=True,
		method='TNC',
		options={'maxiter': 400})

	lambda_ = 1.0
	res2 = optimize.minimize(costFunctionReg,
		initial_theta,
		(X,y,lambda_),
		jac=True,
		method='TNC',
		options={'maxiter': 100})

	# returns the value of costFunction at optimized theta
	cost1 = res1.fun
	cost2 = res2.fun

	# theta is stored in the x property
	theta1 = res1.x
	theta2 = res2.x

	# Create prediction using training data to find the training accuracy
	p = predict(theta1, X)
	print(f'Training accuracy: {np.mean(p == y)}')

	# In general, this regularized prediction should have a higher accuracy
	p = predict(theta2, X)
	print(f'Training accuracy (regularized): {np.mean(p == y)}')


if __name__ == '__main__':
	main()