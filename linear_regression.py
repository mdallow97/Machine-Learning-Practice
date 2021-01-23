# linear_regression.py

'''
This file contains an implementation of linear regression. 
Code based on the project provided by the Coursera, Machine Learning course by Andrew Ng.
'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def generateData(num_features, num_samples, X_range, Y_range):
	def getRandomPoint(r):
		return (float(random.randint((r[0] + 1) * 1000, (r[1] + 1) * 1000)) / 1000.0) - 1.0
	
	X = np.array([[getRandomPoint(X_range) for i in range(num_features)] for j in range(num_samples)])
	y = np.array([[getRandomPoint(Y_range)] for i in range(num_samples)])
	return X,y

def plotData(x, y):
	fig = plt.plot(x, y, 'ro', ms=10, mec='k')
	plt.show()

def featureNormalize(X):
	X_norm = X.copy()
	mu = np.zeros(X.shape[1])
	sigma = np.zeros(X.shape[1])

	for i in range(X.shape[1]):
		mu[i] = np.mean(X[:,i])
		sigma[i] = np.std(X[:,i])

		X_norm[:,i] = (X_norm[:,i] - mu[i]) / sigma[i]

	return X_norm, mu, sigma

def normalizeTestData(X, mu, sigma):
	X_norm = X.copy()
	for i in range(X.shape[1]):
		X_norm[:,i] = (X_norm[:,i] - mu[i]) / sigma[i]

	return X_norm

def computeCost(X, y, theta):
	m = y.size
	J = 0

	for i in range(m):
		H_x = theta.T.dot(X[i])
		J += (H_x - y[i]) ** 2
	J *= (1.0/(2.0 * m))
	return J

def gradientDescent(X, y, theta, alpha=0.1, num_iters=1000):
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
	theta = np.zeros(X.shape[1])
	theta = np.linalg.pinv((np.dot(X.T, X))).dot(X.T).dot(y)
	return theta

def predict(test_data, theta):
	prediction = []
	for sample in test_data:
		prediction.append(theta.T.dot(sample))

	return prediction

def main():
	n,m = 3, 10
	X,y = generateData(n, m, (0,20), (0,10))
	X_norm, mu, sigma = featureNormalize(X)

	# Add intercept term to X
	X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

	# Using gradient descent (iterative)
	theta1 = np.zeros(n+1)
	theta1, J_history = gradientDescent(X, y, theta1)

	# Using normal equation
	theta2 = normalEqn(X, y)

	# Create random test point
	test = normalizeTestData(np.array([[12.213, 18.21, 1.123], [11.72, 5.123, 7.165]]), mu, sigma)
	test = np.concatenate([np.ones((test.shape[0],1)), test], axis=1)

	# Predict given test sample
	prediction1 = predict(test, theta1)
	prediction2 = predict(test, theta2)

	print(prediction1)
	print(prediction2)


if __name__ == '__main__':
	main()