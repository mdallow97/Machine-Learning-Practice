# linear_regression.py

'''
This file contains an implementation of linear regression. 
Code based on the project provided by the Coursera, Machine Learning course by Andrew Ng.
Written in Python rather than Octave or Matlab.
'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import argparse

def generateData(num_features, num_samples, X_range, Y_range):
	"""
	:type num_features: int
	:type num_samples: int
	:type X_range: (int, int)
	:type Y_range: (int, int)
	:rtype: array_like, array_like

	returns random generated data
	X => random training data
	y => random output data

	*** Will likely result in bad predictions because data is random!
	"""

	def getRandomPoint(r):
		return (float(random.randint((r[0] + 1) * 1000, (r[1] + 1) * 1000)) / 1000.0) - 1.0
	
	X = np.array([[getRandomPoint(X_range) for i in range(num_features)] for j in range(num_samples)])
	y = np.array([[getRandomPoint(Y_range)] for i in range(num_samples)])
	return X,y

def plotData(x, y):
	"""
	:type x: array_like
	:type y: array_like

	Plots data points x and y into a figure.
	"""
	fig = plt.plot(x, y, 'ro', ms=10, mec='k')
	plt.show()

def featureNormalize(X):
	"""
	:type X: array_like
	:rtype: array_like, array_like, array_like

	X_norm => the normalized features of X
	mu => array of the means for the sample set of each feature
	sigma => array of the standard deviations for the sample set of each feature
	"""
	X_norm = X.copy()
	mu = np.zeros(X.shape[1])
	sigma = np.zeros(X.shape[1])

	for i in range(X.shape[1]):
		mu[i] = np.mean(X[:,i])
		sigma[i] = np.std(X[:,i])

		# Normalized features = (features - mean) / std deviation
		X_norm[:,i] = (X_norm[:,i] - mu[i]) / sigma[i]

	return X_norm, mu, sigma

def normalizeTestData(X, mu, sigma):
	"""
	:type X: array_like
	:type mu: array_like
	:type sigma:  array_like
	:rtype: array_like

	Uses the regularization params (mu, sigma) retrieved from training data, to normalize a set of test data

	X_norm => the normalized features of X
	"""
	X_norm = X.copy()
	for i in range(X.shape[1]):
		X_norm[:,i] = (X_norm[:,i] - mu[i]) / sigma[i]

	return X_norm

def computeCost(X, y, theta):
	"""
	:type X: array_like
	:type y: array_like
	:type theta: array_like
	:rtype: float

	theta => linear regression parameters
	J => the value of the cost function
	
	Computes the cost for linear regression using the sum of squared error cost function (dependent on theta)
	"""
	m = y.size

	J = 0
	for i in range(m):
		H_x = theta.T.dot(X[i])
		J += (H_x - y[i]) ** 2
	J *= (1.0/(2.0 * m))
	return J

def gradientDescent(X, y, theta, alpha=0.1, num_iters=1000):
	"""
	:type X: array_like
	:type y: array_like
	:type theta: array_like
	:type alpha: float
	:type num_iters: int
	:rtype: array_like, array_like

	alpha => learning rate
	num_iters => number of iterations to run gradient descent
	J_history => keeps track of the cost of each array theta

	Gradient descent uses the cost function to find new values of theta. 
	Updates theta by taking num_iters steps with learning rate alpha.
	With a correct learning rate, we should see that as the algorithm 
	iterates through num_iters, the cost should be decreasing.
	"""
	m = y.shape[0] # number of training examples

	# make a copy of theta, which will be updated by gradient descent
	theta = theta.copy()
	J_history = []

	for i in range(num_iters):
		for j in range(len(theta)):
			der = 0
			for k in range(m):
				der += (theta.T.dot(X[k]) - y[k]) * X[k]
		theta -= (alpha / m) * der

		# save the cost J in every iteration
		J_history.append(computeCost(X, y, theta))

	return theta, J_history

def normalEqn(X, y):
	"""
	:type X: array_like
	:type y: array_like
	:rtype: array_like

	Uses the closed-form solution to linear regression using the normal equation.
	Can be slow when there are a lot of features in X!
	"""
	theta = np.zeros(X.shape[1])
	theta = np.linalg.pinv((np.dot(X.T, X))).dot(X.T).dot(y)
	return theta

def costFunctionReg(X, y, theta, lambda_=1.0):
	"""
	:type X: array_like
	:type y: array_like
	:type theta: array_like
	:type lambda_: float
	:rtype: float

	theta => linear regression parameters
	lambda_ => regularization parameter
	J => the value of the cost function
	
	Similar to computeCost(), but uses a regularization term to reduce magnitude of weights (theta)
	Helps to solve the overfitting issue
	"""
	m = y.size
	n = theta.size

	# J = (1/2m) * ( sum((H(x) - y)^2) + (lambda * ( sum(theta^2))))
	
	# Calculate sum((H(x) - y)^2)
	J = 0
	for i in range(m):
		H_x = theta.T.dot(X[i])
		J += (H_x - y[i]) ** 2

	# Calculate (sum(theta^2))
	reg = 0
	for j in range(n):
		reg += theta[j] ** 2


	J += reg * lambda_ # add regularization term to J
	J *= (1.0/(2.0 * m))
	return J

def gradientDescentReg(X, y, theta, alpha=0.1, num_iters=1000, lambda_=1.0):
	"""
	:type X: array_like
	:type y: array_like
	:type theta: array_like
	:type alpha: float
	:type num_iters: int
	:type lambda_: float
	:rtype: float

	theta => linear regression parameters
	lambda_ => regularization parameter
	J_history => keeps track of the cost of each array theta
	
	Similar to gradientDescent(), but uses a regularization term to reduce magnitude of weights (theta)
	Helps to solve the overfitting issue
	"""

	m = y.shape[0] # number of training examples

	# make a copy of theta, which will be updated by gradient descent
	theta = theta.copy()
	J_history = []

	for i in range(num_iters):
		# theta := theta * (1- (alpha * lambda) / m) - ((alpha / m) * sum((H(x) -  y) * x)
		for j in range(len(theta)):
			der = 0
			for k in range(m):
				der += (theta.T.dot(X[k]) - y[k]) * X[k]
		theta = (theta * (1.0 - ((alpha * lambda_) / m))) - ((alpha / m) * der)

		# save the cost J in every iteration
		J_history.append(costFunctionReg(X, y, theta, lambda_))

	return theta, J_history

def regNormEqn(X, y, lambda_=1.0):
	"""
	:type X: array_like
	:type y: array_like
	:type lambda_: float
	:rtype: array_like

	Similar to normalEqn(), but uses a regularization term to reduce magnitude of weights (theta)
	Helps to solve the overfitting issue
	"""
	theta = np.zeros(X.shape[1])
	L = np.eye(X.shape[1])
	L[0][0] = 0.0
	theta = np.linalg.pinv(X.T.dot(X) + (lambda_ * L)).dot(X.T).dot(y)

	return theta

def predict(test_data, theta):
	"""
	:type test_data: array_like
	:type theta: array_like
	:rtype: array_like

	Uses calculated values of theta to make a prediction given new sample points.
	If the training data was normalized, make sure test data is too!
	"""
	prediction = []
	for sample in test_data:
		prediction.append(theta.T.dot(sample))

	return prediction

def fitLine(X, y, theta, title, step=0.01):
	"""
	:type X: array_like
	:type y: array_like
	:type theta: array_like
	:type title: string
	:type step: float

	Fits a line to the data given theta.
	*** Currently only works for X with one features (but any amount of samples)
	"""
	test = np.array(np.arange(min(X[:,1]), max(X[:,1]), step)).reshape((-1, 1))
	test = np.concatenate([np.ones((test.shape[0],  test.shape[1])), test], axis=1)

	plt.plot(X[:,1], y, 'ro', ms=10, mec='k')
	plt.plot(test[:,1], predict(test, theta), 'b', ms=3, mec='k')
	plt.title(title)
	plt.xlabel("Sample data (X)")
	plt.ylabel("Output data (y)")
	plt.show()

def main():
	# Provide user the option of showing plots
	ap = argparse.ArgumentParser()
	ap.add_argument('--plot', action='store_true', help="Show plots of data when available.")
	args = ap.parse_args()

	# n => number of features, m => sample size
	n,m = 1, 10

	# This data is random; to get a good prediction, use real sample data
	X,y = generateData(n, m, (0,20), (0,10))

	X_norm, mu, sigma = featureNormalize(X)

	# Add intercept term to X
	X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

	# Using gradient descent (iterative)
	theta1 = np.zeros(n+1)
	theta1, J_history1 = gradientDescent(X, y, theta1)

	# Convergence plot for theta1
	if args.plot:
		plt.plot(J_history1)
		plt.title("Convergence plot for gradient descent (iterative)")
		plt.xlabel("Iterations")
		plt.ylabel("Cost (Error)")
		plt.show()

	# Using normal equation
	theta2 = normalEqn(X, y)

	# Using regularized gradient descent (iterative)
	theta3 = np.zeros(n+1)
	theta3, J_history3 = gradientDescentReg(X, y, theta3)

	# Convergence plot for theta3
	if args.plot:
		plt.plot(J_history3)
		plt.title("Convergence plot for regularized gradient descent (iterative)")
		plt.xlabel("Iterations")
		plt.ylabel("Cost (Error)")
		plt.show()

	# Using regularized normal equation
	theta4 = regNormEqn(X, y)

	# Fit line to data (currently only works for input data with one feature (n))
	if args.plot and n == 1:
		fitLine(X, y, theta1, "Line Fit via Iterative Gradient Descent")
		fitLine(X, y, theta2, "Line Fit via Normal Equation")
		fitLine(X, y, theta3, "Line Fit via Iterative Regularized Gradient Descent")
		fitLine(X, y, theta4, "Line Fit via Regularized Normal Equation")



if __name__ == '__main__':
	main()