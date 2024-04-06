import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import random

class InitialPartition:

	def __init__(self, seed = 42):
		self.rs = check_random_state(seed)

	def get_initial_center_index(self, X):
		return random.randint(0, X.shape[0] - 1)

	def select_next_center_index(self, centers_indices, probability_array):
		while(True):
			selected_index = np.random.choice(len(probability_array), p=probability_array)
			if(selected_index not in centers_indices):
				return selected_index

	def calculate_kernel_distances_between_points(self, center_index, centers_indices, X, kernel_matrix, kernel_distances_between_points):
		for i in range(X.shape[0]):
			if i not in centers_indices: 
				kernel_distances_between_points[center_index, i] = kernel_matrix[center_index, center_index] - 2 * kernel_matrix[center_index, i] + kernel_matrix[i, i]
		return kernel_distances_between_points

	def calculate_points_probabilities_to_be_selected(self, kernel_distances_between_points):
		return kernel_distances_between_points / np.sum(kernel_distances_between_points)

	def calculate_minimum_distances_from_centers_and_partition(self, centers_indices, kernel_distances_between_points):
		minimum_distances = np.zeros(kernel_distances_between_points.shape[1])
		partition = np.zeros(kernel_distances_between_points.shape[1], dtype=int)

		for i, column in enumerate(kernel_distances_between_points.T):
			nonzero_indices = np.nonzero(column)[0]
			if len(nonzero_indices) > 0:
				partition[i] = nonzero_indices[np.argmin(column[nonzero_indices])]
				minimum_distances[i] = column[partition[i]]
		
		# Make 0 the distances for centers and set their partition to their index
		minimum_distances[centers_indices] = 0
		partition[centers_indices] = centers_indices
		
		return minimum_distances, partition	

	def calculate_initial_partition_with_kkmeans_pp_initialization(self, K, X, kernel_matrix):
		centers_indices = []
		n_samples = X.shape[0]
		kernel_distances_between_points = np.zeros((n_samples, n_samples))
		
		for i in range(K):
			if i != 0:
				centers_indices.append(self.select_next_center_index(centers_indices, probabilities))
			else:
				centers_indices.append(self.get_initial_center_index(X))
			
			kernel_distances_between_points = self.calculate_kernel_distances_between_points(centers_indices[i], centers_indices, X, kernel_matrix, kernel_distances_between_points)
			minimum_distances, partition = self.calculate_minimum_distances_from_centers_and_partition(centers_indices, kernel_distances_between_points)
			probabilities = self.calculate_points_probabilities_to_be_selected(minimum_distances)
		print(f"\n centers_indices are: {centers_indices}!")
		return centers_indices, partition
	
	def calculate_initial_partition_with_froggy_initialization(self, K, N): 
		partition = self.rs.randint(K, size=N)
		centers_indices = np.unique(partition)
		
		return centers_indices, partition
	
	def calculate_initial_partition(self, K, X, kernel_matrix, method):
		if method == 'Froggy':
			print("Executing Froggy Initialization")
			return self.calculate_initial_partition_with_froggy_initialization(K, X.shape[0])
		
		elif method == 'KMeans':
			print("Executing Froggy Initialization")
			raise Exception("Error! You didn't choose an existing initialization method!")
		
		elif method == 'KMeans++':
			print("Executing Froggy Initialization")
			raise Exception("Error! You didn't choose an existing initialization method!")
		
		elif method == 'KKmeans++':
			print("Executing KKMeans++ Initialization")
			return self.calculate_initial_partition_with_kkmeans_pp_initialization(K, X, kernel_matrix)
		
		else:
			raise Exception("Error! You didn't choose an existing initialization method!") 
