import numpy as np
from numpy.linalg import cholesky  
from numpy.linalg.linalg import LinAlgError

class Modification:

	def __init__(self):
		pass

	# Implement check for positive defined matrix with cholesky
	def check_for_positive_definite_matrix(self, M):
		try:
			cholesky(M)
			return True
		except LinAlgError: 
			# It throws LinAlgError if matrix is not positive defined
			return False
	
	def modify_kernel_matrix(self, M, b):
		# Initialize I = identity matrix and t = 0
		I = np.identity(M.shape[0])
		t = 0
		
		# Find minimum element in diagonal of H
		min_H_ii = np.min(np.diag(M))
		
		# Check if min_H_ii > 0 
		if (min_H_ii > 0):
			t = b
		else:    
			t = -min_H_ii + b
	
		while(True):
			modified_matrix = np.add(M, t * I)
			try:
				# Check for positive defined matrix with cholesky and return it if it is 
				L = cholesky(modified_matrix)
				return modified_matrix
			except LinAlgError:
				# It throws LinAlgError if matrix is not positive defined
				t = max(2 * t, b)
