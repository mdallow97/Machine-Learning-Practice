# linalg.py

"""
This file contains functions commonly seen in linear algebra, and are building blocks to functions used in Machine Learning.
This file was solely for practicing Linear Algebra and making sure I understand the main concepts. It may not scale correctly.
Finally, it does not contain some of the most important functions that are required for machine learning:
- Computing Eigenvalues and Eigenvectors
- Computing gradients
- Computing hessians

Michael Dallow
"""

import numpy as np
import copy

def zeroes(m, n):
	"""
	:type m: int
	:type n: int
	:rtype: List[List[float]]

	returns m x n matrix full of zeroes
	"""
	Z = [[0 for i in range(n)] for j in range(m)]
	assert (np.array(Z) == np.zeros((m,n), dtype=int)).all()
	return Z

def ones(m,n):
	"""
	:type m: int
	:type n: int
	:rtype: List[List[float]]

	returns m x n matrix full of ones
	"""
	Z = [[1 for i in range(n)] for j in range(m)]
	assert (np.array(Z) == np.ones((m,n), dtype=int)).all()
	return Z

def transpose(A):
	"""
	:type n: List[List[float]]
	:rtype: List[List[float]]

	returns the transpose of A
	"""
	t = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
	assert (np.array(t) == np.array(A).T).all()
	return t

def create_identity(n):
	"""
	:type n: int
	:rtype: List[List[float]]

	returns n x n identity matrix
	"""
	I = [[1 if i==j else 0 for i in range(n)] for j in range(n)]
	assert (np.array(I) == np.identity(n)).all()
	return I

def dot_product(A, B):
	"""
	:type A: List[List[float]]
	:type B: List[List[float]]
	:rtype: List[List[float]]
	"""

	A = copy.deepcopy(A)
	B = copy.deepcopy(B)

	# A => (m x n), B => (n x p)
	m,n = len(A), len(A[0])
	p = len(B[0])

	# num cols in A must equal num rows in B
	assert n == len(B)

	# C = AB => (m x p)
	C = zeroes(m, p)
	for i in range(m):
		for j in range(p):
			for k in range(n):
				C[i][j] += A[i][k] * B[k][j]

	assert (np.array(C) == np.dot(A, B)).all()
	return C

def is_symmetric_matrix(A):
	"""
	:type A: List[List[float]]
	:rtype: bool
	"""

	if transpose(A) == A:
		return True
	else:
		return False

def trace(A):
	"""
	:type A: List[List[float]]
	:rtype: int
	"""
	# A must be (n x n)
	n = len(A)
	assert n == len(A[0])

	# the trace is the sum of the diagonal elements
	tr = 0
	for i in range(n):
		tr += A[i][i]

	assert tr == np.trace(A)

	return tr

def L2_norm(x):
	"""
	:type x: List[float]
	:rtype: float
	"""

	# x must be a vector
	assert type(x) is list
	if len(x) == 0:
		return 0
	assert type(x[0]) is int or type(x[0]) is float

	# l2 = sqrt(sum(x[i] ** 2)), i = [0, 1, ..., n]
	l2 = 0
	for i in range(len(x)):
		l2 += x[i]**2

	# sqrt
	l2 **= 0.5
	assert l2 == np.linalg.norm(x)
	return l2

def Lp_norm(x, p):
	"""
	:type x: List[float]
	:type p: float
	:rtype: float
	"""

	# x must be a vector
	assert type(x) is list
	if len(x) == 0:
		return 0
	assert type(x[0]) is int or type(x[0]) is float
	assert p >= 1

	lp = 0
	for i in range(len(x)):
		lp += abs(x[i])**p
	lp **= 1.0 / p

	assert lp == np.linalg.norm(x, ord=p)
	return lp

def frobenius_norm(A):
	"""
	:type A: List[List[float]]
	:rtype: float

	returns l2 norm equivalent for matrix
	"""
	m,n = len(A), len(A[0])

	Af = 0
	for i in range(m):
		for j in range(n):
			Af += A[i][j] ** 2
	Af **= 0.5

	# frobenius norm equals sqrt(trace(A.T * A))
	assert Af == trace(dot_product(transpose(A), A)) ** 0.5
	assert Af == np.linalg.norm(A)
	return Af

def get_row_echelon_matrix(ref, it=0):
	"""
	:type ref: List[List[float]]
	:rtype: List[List[float]]

	returns the row echelon form (REF) of a matrix
	"""
	ref = copy.deepcopy(ref)

	# pivot
	pivot = 0
	for i in range(it, len(ref)):
		if ref[i][it] != 0:
			pivot = ref[i][it]
			break

	# move pivot row to top
	temp = ref[i]
	ref.pop(i)
	ref = ref[0:it] + [temp] + ref[it:len(ref)]

	# if pivot is 0, either reached end of matrix or column is all 0's (below prev pivot row)
	if pivot == 0:
		if it == len(ref) - 1:
			return ref
		else:
			it += 1
			return get_row_echelon_matrix(ref, it)
	
	# make pivot equal one
	for j in range(len(ref[it])):
		ref[it][j] /= pivot

	# reduce proceding rows given pivot row
	for i in range(it+1, len(ref)):
		if ref[i][it] != 0:
			multiplier = ref[i][it]
			for j in range(it, len(ref[i])):
				ref[i][j] -= multiplier * ref[it][j]

	if it == len(ref) - 1:
		return ref
	else:
		it += 1
		return get_row_echelon_matrix(ref, it)

def rank(A):
	"""
	:type A: List[List[float]]
	:rtype: int

	returns number of linearly independent column vectors
	"""
	ref = get_row_echelon_matrix(A)
	m,n = len(A), len(A[0])
	z = zeroes(1, n)[0]
	
	r = 0
	for row in ref:
		if row != z:
			r += 1

	assert r == np.linalg.matrix_rank(A)

	return r

def determinant(A, debug=False):
	"""
	:type A: List[List[float]]
	:type debug: bool
	:rtype: float

	returns |A|
	"""
	def determinant_helper(A):
		m,n = len(A), len(A[0])

		assert m == n

		if n == 1:
			return A[0][0]
		elif n == 2:
			# definition of the determinant for a 2 x 2 matrix
			return A[0][0] * A[1][1] - A[0][1] * A[1][0]
		"""
		|A| = sum( ((-1) ^ i) * A[0][i] * |A[!0][!i]| )
		A[!0][!i] is the matrix without row 0 and column i
		"""
		det = 0
		for i in range(n):
			M = get_inner_matrix(A, 0, i)
			det += ((-1) ** i) * A[0][i] * determinant_helper(M)

		return det

	det = determinant_helper(A)
	if det != np.linalg.det(A) and debug:
		# Likely not exactly the same
		print(f"WARNING: Determinant's ({str(det)}) not equal to NumPy ({str(np.linalg.det(A))})")
	return det

def adjoint(A):
	"""
	:type A: List[List[float]]
	:rtype: List[List[float]]

	returns adj(A)
	"""

	m,n = len(A), len(A[0])
	assert m == n and n > 1

	# find cofactor matrix
	cofactor = ones(m, n)
	for i in range(n):
		for j in range(n):
			M = get_inner_matrix(A, i, j)
			cofactor[i][j] *= ((-1) ** (i+j)) * determinant(M)

	# adjoint (adjugate) matrix is transpose of cofactor matrix
	return transpose(cofactor)

def inverse(A):
	"""
	:type A: List[List[float]]
	:rtype: List[List[float]]

	returns A^(-1)
	"""
	m,n = len(A), len(A[0])
	assert m == n

	det = determinant(A)
	adj = adjoint(A)

	if det == 0:
		print("WARNING: Matrix is non-invertible")
		return zeroes(m,n)

	inv = [[(1 / det) * adj[i][j] for i in range(n)] for j in range(n)]

	if (np.array(inv) != np.linalg.inv(A)).all():
		print(f"WARNING: Inverse is not equal to NumPy\nInverse:\n{np.array(inv)}\nNumPy Inverse:\n{np.linalg.inv(A)}")

	return inv

def is_orthogonal(A):
	"""
	:type A: List[List[float]]
	:rtype: bool
	"""
	return transpose(A) == inverse(A)

def get_inner_matrix(A, i, j):
	"""
	:type A: List[List[float]]
	:rtype: List[List[float]]

	returns the matrix A with row i and column j removed
	"""
	M = copy.deepcopy(A)
	[k.pop(j) for k in M]
	M.pop(i)
	return M



def main():
	A = [[-2,-4,2], [-2,1,2], [4,2,5]]
	print(np.array(get_row_echelon_matrix(A)))
	

if __name__ == "__main__":
	main()








