import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import random
import math

class InitialPartition:

	def __init__(self, seed = 42):
		self.rs = check_random_state(seed)

	def calculate_clusters_labels(self, K):
		return np.arange(K)

	def get_initial_center_index(self, N):
		return random.randint(0, N - 1)

	def select_next_center_index(self, centers_indices, probability_array):
		while(True):
			selected_index = np.random.choice(len(probability_array), p=probability_array)
			
			if(selected_index not in centers_indices):
				return selected_index

	def calculate_kernel_distance_between_points(self, i, j, kernel_matrix):
		return kernel_matrix[i, i] - (2 * kernel_matrix[i, j]) + kernel_matrix[j, j]
	
	def calculate_kernel_distances_between_points(self, center_index, centers_indices, kernel_matrix, kernel_distances_between_points):
		for i in range(kernel_matrix.shape[0]):
			
			if i not in centers_indices:
				kernel_distances_between_points[center_index, i] = self.calculate_kernel_distance_between_points(center_index, i, kernel_matrix)
		
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

	def scale_partition(self, K, partition):
		partition = LabelEncoder().fit_transform(partition)
		clusters_labels = self.calculate_clusters_labels(K)

		return clusters_labels, partition

	def calculate_initial_partition_with_kkmeans_pp_initialization(self, K, X, kernel_matrix):
		centers_indices = []
		N = X.shape[0]
		kernel_distances_between_points = np.zeros((N, N))

		for i in range(K):
			if i != 0:
				centers_indices.append(self.select_next_center_index(centers_indices, probabilities))
			else:
				centers_indices.append(self.get_initial_center_index(N))

			kernel_distances_between_points = self.calculate_kernel_distances_between_points(centers_indices[i], centers_indices, kernel_matrix, kernel_distances_between_points)
			minimum_distances, partition = self.calculate_minimum_distances_from_centers_and_partition(centers_indices, kernel_distances_between_points)
			probabilities = self.calculate_points_probabilities_to_be_selected(minimum_distances)

		return self.scale_partition(K, partition)

	def calculate_initial_partition_with_forgy_initialization(self, K, N):
		partition = self.rs.randint(K, size=N)
		
		return self.scale_partition(K, partition)

	def select_random_integers_without_replacement(self, K, N):
		return random.sample(range(N), K)

	def calculate_euclidean_distance(self, x1, x2):
		return np.sqrt(np.sum((x1 - x2) ** 2))

	def calculate_distance(self, X, i, j, kernel_matrix, method):
		if method == 'kMeans':
			return self.calculate_euclidean_distance(X[i], X[j])
		
		elif method == 'KkMeans':
			return self.calculate_kernel_distance_between_points(i, j, kernel_matrix)
		
		else:
			raise Exception("Error! You didn't choose an existing calculate distance method!")

	def calculate_point_cluster_assignment(self, K, X, i, centers_indices, kernel_matrix, method):
		min_distance = math.inf
		cluster_assignment = -1

		for j in range(K):
			cur_distance = self.calculate_distance(X, i, centers_indices[j], kernel_matrix, method)

			if(cur_distance < min_distance):
				min_distance = cur_distance
				cluster_assignment = j

		return cluster_assignment 		

	def calculate_initial_partition_with_kmeans_initialization(self, K, X, kernel_matrix, method='kMeans'):
		N = X.shape[0]
		centers_indices = self.select_random_integers_without_replacement(K, N)
		partition = np.zeros(N)

		for i in range(N):
			partition[i] = self.calculate_point_cluster_assignment(K, X, i, centers_indices, kernel_matrix, method)

		return self.scale_partition(K, partition)

	def calculate_initial_partition(self, K, X, kernel_matrix, method):
		if method == 'Forgy':
			print("Executing Forgy Initialization")
			return self.calculate_initial_partition_with_forgy_initialization(K, X.shape[0])

		elif method == 'kMeans':
			print("Executing kMeans Initialization")
			return self.calculate_initial_partition_with_kmeans_initialization(K, X, kernel_matrix, method)

		elif method == 'KkMeans':
			print("Executing KkMeans Initialization")
			return self.calculate_initial_partition_with_kmeans_initialization(K, X, kernel_matrix, method)	

		elif method == 'kMeans++':
			print("Executing Forgy Initialization")
			raise Exception("Error! You didn't choose an existing initialization method!")

		elif method == 'KkMeans++':
			print("Executing KkMeans++ Initialization")
			return self.calculate_initial_partition_with_kkmeans_pp_initialization(K, X, kernel_matrix)

		else:
			raise Exception("Error! You didn't choose an existing initialization method!")
