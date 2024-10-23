import numpy as np
from numpy.linalg import cholesky  
from numpy.linalg.linalg import LinAlgError

class Modification:

	def __init__(self):
		pass

	def modify_kernel_matrix_ra(self, A, b=0.1):
		# Initialize I = identity matrix and t = 0
		I = np.identity(A.shape[0])
		t = 0

		while(True):
			modified_matrix = np.add(A, t * I)
			try:
				# Check for positive defined matrix with cholesky and return it if it is 
				L = cholesky(modified_matrix)
				return modified_matrix
			except LinAlgError:
				# It throws LinAlgError if matrix is not positive defined
				t = max(2 * t, b)

	def modify_kernel_matrix_nc(self, A, D, b=0.1):
		# Initialize I = identity matrix, K and t = 0
		D_inv = np.linalg.inv(D)
		K = D_inv @ A @ D_inv
		t = 0

		while(True):
			modified_matrix = np.add(K, t * D_inv)
			try:
				# Check for positive defined matrix with cholesky and return it if it is 
				L = cholesky(modified_matrix)
				return modified_matrix
			except LinAlgError:
				# It throws LinAlgError if matrix is not positive defined
				t = max(2 * t, b)
			
