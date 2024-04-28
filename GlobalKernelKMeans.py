from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from KernelKMeans import KernelKMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

class _BaseGlobalKernelKMeans(BaseEstimator, ClusterMixin, TransformerMixin, ABC):
	"""Base class for Global K-Means, Global K-Means++ and future (or past) variants.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			tol (float) : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			cluster_centers_ (dict) : Dictionary to store cluster centers for each sub-problem k.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
	"""
	def __init__(self, n_clusters, kernel_matrix, tol, verbose):
		self.n_clusters = n_clusters
		self.kernel_matrix = kernel_matrix
		self.tol = tol
		self.verbose = verbose
		
		self.n_iter_ = 0
		#self.cluster_centers_ = {} WHAT TO DO WITH THEM? Maybe Remove them
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


class GlobalKernelKMeans(_BaseGlobalKernelKMeans):
	"""Global K-Means clustering algorithm.

		Parameters:
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			tol (float) : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			cluster_centers_ (dict) : Dictionary to store cluster centers for each sub-problem k.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.

	"""
	def __init__(self, n_clusters=8, kernel_matrix=None, tol=1e-4, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			tol=tol,
			verbose=verbose,
		)
	
	def fit(self, X, y=None, sample_weight=None):
		"""Compute the global k-means clustering.

		Parameters
		----------
			X (array-like) : Input data.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self : Fitted estimator.
		"""
		check_array(X)
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving Kernel {k}-means')
						
			self.inertia_[k] = float('inf')
			for i, xi in enumerate(X): # TODO parallel
				kernelKMeans = KernelKMeans(n_clusters=k, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=None).fit(X)
				self.n_iter_ += kernelKMeans.n_iter_
				
				if kernelKMeans.inertia_ < self.inertia_[k]:
					#self.cluster_centers_[k] = kernelKMeans.cluster_centers_
					self.labels_[k] = kernelKMeans.labels_
					self.inertia_[k] = kernelKMeans.inertia_

		return self
