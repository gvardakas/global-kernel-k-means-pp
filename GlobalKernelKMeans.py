from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import pairwise_distances
from KernelKMeans import KernelKMeans

class _BaseGlobalKernelKMeans(BaseEstimator, ClusterMixin, TransformerMixin, ABC):
	"""Base class for Global Kernel K-Means, Global Kernel K-Means++ and future (or past) variants.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			kernel_matrix (numpy.array) : The kernel matrix array.
			verbose (int) : Verbosity mode.
			N (int) : The number of points.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
	"""
	def __init__(self, n_clusters, kernel_matrix, verbose):
		self.n_clusters = n_clusters
		self.kernel_matrix = kernel_matrix
		self.verbose = verbose
		self.N = kernel_matrix.shape[0]
		
		self.n_iter_ = 0
		self.labels_ = {}
		self.inertia_ = {}

	@abstractmethod	
	def fit(self, X=None, y=None, sample_weight=None):
		"""Abstract method for fitting the model to the data.

		Parameters
		----------
			X (array-like) : Ignored.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self: Fitted estimator.
		"""
		return self

	def predict(self, X=None):
		"""Predict cluster labels for data X with n_clusters clusters.

		Parameters
		----------
			X : Ignored. Models should already be fitted.

		Returns
		----------
			labels_ (array) : Cluster labels.
		"""
		return self.labels_[self.n_clusters]

	def fit_predict(self, X=None, y=None, sample_weight=None):
		"""Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).


		Parameters
		----------
			X (array-like) : Ignored.

		Returns
		----------
			labels_ (array) : Cluster labels.
		"""
		self.fit(X, y, sample_weight)
		return self.predict(X)

class GlobalKernelKMeans(_BaseGlobalKernelKMeans):
	"""Global Kernel K-Means clustering algorithm.

		Parameters:
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			kernel_matrix (numpy.array) : The kernel matrix array.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.

	"""
	def __init__(self, n_clusters=8, kernel_matrix=None, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			verbose=verbose,
		)

	
	def fit(self, X=None, y=None, sample_weight=None):
		"""Compute the global kernel k-means clustering.

		Parameters
		----------
			X (array-like) : Input data.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self : Fitted estimator.
		"""
		initial_labels_ = np.zeros(self.N)
		
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving Kernel {k}-means')
						
			self.inertia_[k] = float('inf')
			for i in range(self.N):
				if(np.where(initial_labels_ == initial_labels_[i])[0].shape[0] <= 1): # Check for clusters with 1 point
					continue

				prev_xi_label = initial_labels_[i]
				initial_labels_[i] = k-1

				kernelKMeans = KernelKMeans(n_clusters=k, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=np.copy(initial_labels_), verbose=0).fit()
				self.n_iter_ += kernelKMeans.n_iter_
				
				initial_labels_[i] = prev_xi_label

				if kernelKMeans.inertia_ < self.inertia_[k]:
					self.labels_[k] = kernelKMeans.labels_
					self.inertia_[k] = kernelKMeans.inertia_
		
			initial_labels_ = self.labels_[k]

			if self.verbose > 0: 
				print(f'Solved {k}-means MSE: {self.inertia_[k]}')
		
		return self

class GlobalKernelKMeansPP(_BaseGlobalKernelKMeans):
	"""Global Kernel K-Means++ clustering algorithm.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			kernel_matrix (numpy.array) : The kernel matrix array.
			n_candidates (int) : The number of centroid candidates to examine.
			sampling (str): The sampling method utilized for the centroid candidates ('batch' or 'sequential').
			verbose (int) : Verbosity mode.
				
		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
			cluster_distance_space_(dict) : Dictionary to store the distance of each data point to the nearest centroid.
	"""
	def __init__(self, n_clusters=8, kernel_matrix=None, n_candidates=25, sampling='batch', verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			verbose=verbose,
		)
		self.n_candidates = n_candidates
		self.sampling = sampling
		
		self.cluster_distance_space_ = dict()
	
	def fit(self, X=None, y=None, sample_weight=None):
		"""Compute the global kernel k-means++ clustering.

		Parameters
		----------
			X (array-like) : Ignored.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self : Fitted estimator.
		"""
		initial_labels_ = np.zeros(self.N)
		kernelKMeans = KernelKMeans(n_clusters=1, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=initial_labels_).fit()
		self.labels_[1] = kernelKMeans.labels_
		self.inertia_[1] = kernelKMeans.inertia_
		self.cluster_distance_space_[1] = kernelKMeans.min_distances
		
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving {k}-means')
				
			centroid_candidates = self._sampling(self.cluster_distance_space_[k-1])
			self.inertia_[k] = float('inf')
			for i in centroid_candidates:
				prev_xi_label = initial_labels_[i]
				initial_labels_[i] = k-1
				
				kernelKMeans = KernelKMeans(n_clusters=k, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=initial_labels_, verbose=0).fit()
				self.n_iter_ += kernelKMeans.n_iter_
				initial_labels_[i] = prev_xi_label
				
				if kernelKMeans.inertia_ < self.inertia_[k]:
					self.labels_[k] = kernelKMeans.labels_
					self.inertia_[k] = kernelKMeans.inertia_
					self.cluster_distance_space_[k] = kernelKMeans.min_distances

			initial_labels_ = self.labels_[k]
			
			if self.verbose > 0: 
				print(f'Solved {k}-means MSE: {self.inertia_[k]}')
		
		return self
	
	def _sampling(self, cluster_distance_space):
		if self.sampling == 'batch':
			return self._kernel_kmeans_pp_batch(cluster_distance_space)
		elif self.sampling == 'sequential': # TODO FIX IT
			return self._kernel_kmeans_pp_sequential(cluster_distance_space)
		else:
			raise ValueError("Wrong sampling method! options = ['batch', 'sequential']")
	
	def _kernel_kmeans_pp_batch(self, cluster_distance_space):
		cluster_distance_space = np.power(cluster_distance_space, 2).flatten()
		sum_distance = np.sum(cluster_distance_space)
		selection_prob = cluster_distance_space / sum_distance
		selected_indexes = np.random.choice(self.N, size=self.n_candidates, p=selection_prob, replace=False) 
		return selected_indexes
	
	def calculate_kernel_distance_between_points(self, i, j):
		return self.kernel_matrix[i, i] - (2 * self.kernel_matrix[i, j]) + self.kernel_matrix[j, j]
	
	def _kernel_kmeans_pp_sequential(self, cluster_distance_space):
		selected_indexes = np.zeros(shape=(self.n_candidates), dtype=np.int32)       
		cluster_distance_space = cluster_distance_space.flatten()
		for i in range(self.n_candidates):
			cluster_distance_space_squared = np.power(cluster_distance_space, 2)
			sum_distance = np.sum(cluster_distance_space_squared)
			selection_prob = cluster_distance_space_squared / sum_distance
			selected_indexes[i] = np.random.choice(self.N, size=1, p=selection_prob, replace=False)
			
			# OLD CODE WITH X
			#candidate = X[selected_indexes[i]]
			#candidate = np.reshape(candidate, newshape=(1, candidate.shape[0]))
			#candidate_data_distances = pairwise_distances(candidate, X).flatten()
			
			# NEW CODE WITHOUT X
			# Question Do we want distances Distances in Kernel Space so we will use Kii + Kjj - 2*Kii*Kjj?  
			candidate_data_distances = np.zeros(self.N)
			for j in range(self.N):
				candidate_data_distances[j] = self.calculate_kernel_distance_between_points(selected_indexes[i],j)

			# Update probability distribution
			for i, _ in enumerate(cluster_distance_space):
				if candidate_data_distances[i] < cluster_distance_space[i]:
					cluster_distance_space[i] = candidate_data_distances[i]
		
		return selected_indexes