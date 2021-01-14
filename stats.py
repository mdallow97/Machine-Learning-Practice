# stats.py

"""
This file contains functions commonly seen in statistics, and are building blocks to functions used in Machine Learning.
This file was solely for practicing statistics and making sure I understand the main concepts. It may not scale correctly.
Furthermore, all functions here assume DISCRETE probability distributions.

Michael Dallow
"""

import numpy as np
import copy

def joint_prob(p_known, p_cond):
	"""
	:type p_known: float
	:type p_cond: float
	:rtype: float

	returns joint probability P(A) * P(B|A) == P(A,B)
	"""
	return p_known * p_cond

def cond_prob(p_AB, p_B):
	"""
	:type p_AB: float
	:type P_B: float
	:rtype: float

	returns conditional probability P(A|B) == P(A,B) / P(B)
	"""
	return p_AB / p_B

def total_prob(p_A, p_B_A):
	"""
	:type p_A: List[float]
	:type p_B_A: List[float]
	:rtype: float

	returns total probability => sum(P(A[i]) * p_B_A[i])
	"""
	assert len(p_A) == len(p_B_A)

	total = 0
	for i in range(len(p_A)):
		total += joint_prob(p_A[i], p_B_A[i])

	return total

def bayes_rule(p_a, p_B_a, p_B):
	"""
	:type p_a: float
	:type p_B_a: float
	:type p_B: float
	:rtype: float

	returns P(A[i]|B)
	"""
	return cond_prob(joint_prob(p_a, p_B_a), p_B)

def expectation(X, p_X=[], exp=1):
	"""
	:type X: List[float]
	:type p_X: List[float]
	:type exp: float
	:rtype: float

	returns expected value of X
	"""
	E = 0
	if len(p_X) > 0:
		assert len(X) == len(p_X)
		for i in range(len(X)):
			E += (X[i] ** exp) * p_X[i]
	else:
		E = sum(X) / len(X)

	return E

def variance(X, p_X=[]):
	"""
	:type X: List[float]
	:type p_X: List[float]
	:rtype: float

	returns variance of X
	"""
	return expectation(X, p_X, exp=2) - (expectation(X, p_X) ** 2)

def std_deviation(X, p_X):
	"""
	:type X: List[float]
	:type p_X: List[float]
	:rtype: float

	returns standard deviation of X
	"""
	return (variance(X, p_X) ** 0.5)

def covariance(X, Y, p_X=[], p_Y=[]):
	"""
	:type X: List[float]
	:type Y: List[float]
	:type p_X: List[float]
	:type p_Y: List[float]
	:rtype: float

	returns covariance of X and Y => Cov(X,Y)
	"""
	assert len(X) == len(Y)

	E_X = expectation(X, p_X)
	E_Y = expectation(Y, p_Y)

	E_XY = 0
	for i in range(len(X)):
		E_XY += (X[i] - E_X) * (Y[i] - E_Y)

	return E_XY / (len(X) - 1)

def correlation(X,Y, p_X=[], p_Y=[]):
	"""
	:type X: List[float]
	:type Y: List[float]
	:type p_X: List[float]
	:type p_Y: List[float]
	:rtype: float

	returns correlation of X and Y => Cov(X,Y)
	"""
	return covariance(X, Y, p_X, p_Y) / (std_deviation(X, p_X) * std_deviation(Y, p_Y))

def main():
	X = [i for i in range(1,10)]
	Y = X[::-1]
	print(X, Y)
	print(covariance(X, Y))
	

if __name__ == "__main__":
	main()








