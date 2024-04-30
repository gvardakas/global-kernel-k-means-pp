from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import math
from Initialization import Initialization

class _BaseKernelKMeans(BaseEstimator, ClusterMixin, TransformerMixin, ABC):
	"""Base class for Kernel K-Means, Kernel K-Means++ and future (or past) variants.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			kernel_matrix (numpy.array) : The kernel matrix array.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
	"""
	def __init__(self, n_clusters, kernel_matrix, n_init, init, initial_labels_, verbose):
		self.n_clusters = n_clusters
		self.kernel_matrix = kernel_matrix
		self.n_init = n_init
		self.init = init
		self.initial_labels_=initial_labels_
		self.verbose = verbose

		self.initialization = Initialization()
		self.n_iter_ = 0
		self.labels_ = {}
		self.inertia_ = {}

	@abstractmethod	
	def fit(self, X, y=None, sample_weight=None):
		"""Abstract method for fitting the model to the data.

		Parameters
		----------
			X (array-like) : Input data.
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

	def fit_predict(self, X, y=None, sample_weight=None):
		"""Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).


		Parameters
		----------
			X (array-like) : Input data.

		Returns
		----------
			labels_ (array) : Cluster labels.
		"""
		self.fit(X, y, sample_weight)
		return self.predict(X)

class KernelKMeans(_BaseKernelKMeans):

    def __init__(self, n_clusters, kernel_matrix, n_init=10, init='k-means++', initial_labels_=None, verbose=0):
        super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			n_init=n_init,
			init=init,
			initial_labels_=initial_labels_,
			verbose=verbose,
		)
    
    def calculate_ground_truth_error(self, labels_):
        clusters_identities, labels_ = self.initialization.scale_partition(self.n_clusters, labels_)
        N = len(labels_)
        kernel_diag = np.diag(self.kernel_matrix)
        distances = np.zeros((self.n_clusters, N))
        
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels_ == clusters_identities[i])[0]

            n_cluster_samples = len(cluster_indices)
            stable_sum = np.sum(self.kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
            sample_sums = np.sum(self.kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

            distances[i] = kernel_diag - 2 * sample_sums + stable_sum

        min_distances = np.min(distances, axis=0)
        
        return np.sum(min_distances)

    def kernel_kmeans_functionallity(self, N, initial_labels_, kernel_matrix):
        clusters_identities, initial_labels_ = self.initialization.scale_partition(self.n_clusters, initial_labels_)
        distances = np.zeros((self.n_clusters, N))
        previous_labels_ = initial_labels_
        
        kernel_diag = np.diag(kernel_matrix)
        self.n_iter_ = 0
        
        while True:
                distances = np.zeros((self.n_clusters, N))
                for i in range(self.n_clusters):
                    cluster_indices = np.where(previous_labels_ == clusters_identities[i])[0]
                    
                    n_cluster_samples = len(cluster_indices)
                    stable_sum = np.sum(kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
                    sample_sums = np.sum(kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

                    distances[i] = kernel_diag - 2 * sample_sums + stable_sum

                self.min_distances = np.min(distances, axis=0)
                inertia_ = np.sum(self.min_distances)

                current_labels_ = np.argmin(distances, axis=0)
                are_equal = np.array_equal(previous_labels_, current_labels_)

                if are_equal:
                    #print(f'Finished in Iter: {self.n_iter_} Cl L: {inertia_:.4f}')
                    return current_labels_, inertia_

                #print(f'Iter: {self.n_iter_} Cl L: {inertia_:.4f}')
                previous_labels_ = current_labels_
                self.n_iter_ += 1

    def fit(self, X):
        self.inertia_ = math.inf
        self.labels_ = []
        N = X.shape[0]
        
        for _ in range(self.n_init):
            if(self.initial_labels_ is None):    
                self.initial_labels_ = self.initialization.calculate_initial_partition(self.n_clusters, X, self.kernel_matrix, self.init)

            current_labels_, current_inertia_ = self.kernel_kmeans_functionallity(N, initial_labels_ = self.initial_labels_, kernel_matrix=self.kernel_matrix)
            
            self.initial_labels_ = None

            if(current_inertia_ < self.inertia_):
                self.inertia_ = current_inertia_
                self.labels_ = current_labels_
        
        return self        