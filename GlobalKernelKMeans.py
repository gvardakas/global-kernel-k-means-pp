from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import pairwise_distances
import time
from KernelKMeans import KernelKMeans
from Common_Modules.General_Functions import General_Functions
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
	def __init__(self, n_clusters, kernel_matrix, data_dir_path, verbose):
		self.n_clusters = n_clusters
		self.kernel_matrix = kernel_matrix
		self.verbose = verbose
		self.data_dir_path = data_dir_path
		self.N = kernel_matrix.shape[0]
		
		self.n_iter_ = 0
		self.labels_ = {}
		self.inertia_ = {}
		self.execution_times_ = {}
		self.n_iters_ = {}

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
	def __init__(self, n_clusters=8, kernel_matrix=None, data_dir_path=None, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			data_dir_path=data_dir_path,
			verbose=verbose
		)

	
	def fit(self, X=None, y=None, sample_weights=None):
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

			start_time = time.time()			
			self.n_iter_ = 0
			self.inertia_[k] = float('inf')
			for i in range(self.N):
				if(np.where(initial_labels_ == initial_labels_[i])[0].shape[0] <= 1): # Check for clusters with 1 point
					continue

				prev_xi_label = initial_labels_[i]
				initial_labels_[i] = k-1

				kernelKMeans = KernelKMeans(n_clusters=k, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=np.copy(initial_labels_), verbose=0).fit(sample_weights=sample_weights)
				self.n_iter_ += kernelKMeans.n_iter_
				
				initial_labels_[i] = prev_xi_label

				if kernelKMeans.inertia_ < self.inertia_[k]:
					self.labels_[k] = kernelKMeans.labels_
					self.inertia_[k] = kernelKMeans.inertia_
		
			initial_labels_ = self.labels_[k]
			self.execution_times_[k] = time.time() - start_time
			self.n_iters_[k] = self.n_iter_
			
			new_row = { "K": k, "MSE": self.inertia_[k], "ITERATIONS": self.n_iters_[k], "EXECUTION TIME": self.execution_times_[k]}
			General_Functions.append_to_csv(self.data_dir_path, new_row)

			if self.verbose > 0: 
				print(f'Solved {k}-means MSE: {self.inertia_[k]} in {self.execution_times_[k]}s')
		
		if self.verbose > 0: 
				print(f'Total execution time was {sum(self.execution_times_.values())}s')
		
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
	def __init__(self, n_clusters=8, kernel_matrix=None, n_candidates=25, sampling='batch', data_dir_path=None, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			kernel_matrix=kernel_matrix,
			verbose=verbose,
			data_dir_path=data_dir_path
		)
		self.n_candidates = n_candidates
		self.sampling = sampling
		
		self.cluster_distance_space_ = dict()
	
	def fit(self, X=None, y=None, sample_weights=None):
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
		kernelKMeans = KernelKMeans(n_clusters=1, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=initial_labels_).fit(sample_weights=sample_weights)
		self.labels_[1] = kernelKMeans.labels_
		self.inertia_[1] = kernelKMeans.inertia_
		self.cluster_distance_space_[1] = kernelKMeans.min_distances
		
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving {k}-means')

			start_time = time.time()
			self.n_iter_ = 0
			centroid_candidates = self._sampling(self.cluster_distance_space_[k-1])
			self.inertia_[k] = float('inf')
			
			winner_i = -1	
			for winner_index, i in enumerate(centroid_candidates):
				prev_xi_label = initial_labels_[i]
				initial_labels_[i] = k-1
				
				kernelKMeans = KernelKMeans(n_clusters=k, kernel_matrix=self.kernel_matrix, n_init=1, initial_labels_=initial_labels_, verbose=0).fit(sample_weights=sample_weights)
				self.n_iter_ += kernelKMeans.n_iter_
				initial_labels_[i] = prev_xi_label
				
				if kernelKMeans.inertia_ < self.inertia_[k]:
					winner_i = winner_index
					self.labels_[k] = kernelKMeans.labels_
					self.inertia_[k] = kernelKMeans.inertia_
					self.cluster_distance_space_[k] = kernelKMeans.min_distances
			
			#self.plot_solution(X, self.labels_[k-1], centroid_candidates, k, winner_i)
			
			initial_labels_ = self.labels_[k]
			self.execution_times_[k] = time.time() - start_time
			self.n_iters_[k] = self.n_iter_
			new_row = { "K": k, "MSE": self.inertia_[k], "ITERATIONS": self.n_iters_[k], "EXECUTION TIME": self.execution_times_[k]}
			General_Functions.append_to_csv(self.data_dir_path, new_row)
			 
			if self.verbose > 0: 
				print(f'Solved {k}-means MSE: {self.inertia_[k]} in {self.execution_times_[k]}s')
		
		if self.verbose > 0: 
				print(f'Total execution time was {sum(self.execution_times_.values())}s')
		
		return self
	
	def _sampling(self, cluster_distance_space):
		if self.sampling == 'batch':
			return self._kernel_kmeans_pp_batch(cluster_distance_space)
		elif self.sampling == 'sequential':
			return self._kernel_kmeans_pp_sequential(cluster_distance_space)
		else:
			raise ValueError("Wrong sampling method! options = ['batch', 'sequential']")
	
	def _kernel_kmeans_pp_batch(self, cluster_distance_space):
		cluster_distance_space = np.power(cluster_distance_space, 2).flatten()
		sum_distance = np.sum(cluster_distance_space)
		self.selection_prob = cluster_distance_space / sum_distance
		selected_indexes = np.random.choice(self.N, size=self.n_candidates, p=self.selection_prob, replace=False) 
		return selected_indexes
	
	def _kernel_kmeans_pp_sequential(self, cluster_distance_space):
		candidate_data_distances = np.zeros(self.N)
		selected_indexes = np.zeros(shape=(self.n_candidates), dtype=np.int32)       
		cluster_distance_space = cluster_distance_space.flatten()
		for i in range(self.n_candidates):
			cluster_distance_space_squared = np.power(cluster_distance_space, 2)
			sum_distance = np.sum(cluster_distance_space_squared)
			self.selection_prob = cluster_distance_space_squared / sum_distance
			selected_indexes[i] = np.random.choice(self.N, size=1, p=self.selection_prob, replace=False)
			
			candidate_index = selected_indexes[i]
			for j in range(self.N):
				candidate_data_distances[j] = self.kernel_matrix[candidate_index, candidate_index] - (2 * self.kernel_matrix[candidate_index, j]) + self.kernel_matrix[j, j]

			# Update probability distribution
			for i, _ in enumerate(cluster_distance_space):
				if candidate_data_distances[i] < cluster_distance_space[i]:
					cluster_distance_space[i] = candidate_data_distances[i]
		
		return selected_indexes

	def plot_solution(self, X, kmeans_labels, candidates, K, winner_index):
		color_list = [
            "#008897",
            "#9400D3",  # Dark Violet
            "#FFB347",  # Papaya Orange
            "#00FF00",  # Lime Green
            "#FF1493",  # Deep Pink
            "#00FFFF",  # Cyan,
			"#FF4500",  # Orange Red
            "#a13830",
			"#e0cd4d",
			"#101010",  # Rich Black
			"#40E0D0",  # Turquoise,
			"#00FA9A",  # Medium Spring Green
            "#6b6a72",
            "#0000FF",  # Pure Blue
            "#7c503a",
			"#7c4e75",
			"#228B22",  # Forest Green
            "#9B870C",  # Dark Yellow
        ]

		# Create a figure with subplots
		fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

		# Extract candidate points from the dataset
		X_candidates = X[candidates]

		for axis in ['top', 'bottom', 'left', 'right']:
			ax1.spines[axis].set_linewidth(2.0)

		# Get unique labels from kmeans_labels
		unique_labels = np.unique(kmeans_labels)

		# Plot each cluster with a different color
		for label in unique_labels:
			# Get the data points for the current cluster
			cluster_points = X[kmeans_labels == label]
			# Determine the color for this cluster
			color = color_list[label % len(color_list)]  # Cycle through color_list if needed
			# Scatter plot for the cluster points
			ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', 
						color=color, edgecolor='none', s=50, alpha=0.7)
		
		X_candidates_other = np.delete(X_candidates, winner_index, axis=0)
		
		ax1.scatter(X_candidates_other[:,0], X_candidates_other[:,1], c='lime', marker='P', s=400, edgecolors='black')
		ax1.scatter(X_candidates[winner_index,0], X_candidates[winner_index,1], c='red', marker='*', s=500, edgecolors='black')
		
		# Legend
		leg_data = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Data Instance', markeredgecolor='black', markersize=15)
		leg_winner_candidate = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', label='Winner Candidate', markeredgecolor='black', markersize=15)
		leg_candidates = Line2D([0], [0], marker='P', color='w', markerfacecolor='lime', label='Selected Candidate', markeredgecolor='black', markersize=15)
		legend = ax1.legend(handles=[leg_data, leg_winner_candidate, leg_candidates], loc='upper center', framealpha=0.8, fontsize=15)

		legend.get_frame().set_linewidth(2.0)
		legend.get_frame().set_edgecolor('black')

		# Remove axis ticks for both subplots
		#for ax in (ax2):
		ax1.set_xticks([])
		ax1.set_yticks([])

		# Adjust layout and save the plot
		plt.tight_layout()
		#plt.savefig(f"probabilities_and_clustering_{K - 1}_{self.sampling}.png")
		plt.savefig(f"g{K - 1}_pp_{self.sampling}.png")
		plt.close()

	def plot_solution_3(self, X, kmeans_labels, candidates, K, winner_index):
		color_list = [
            "#008897",
            "#9400D3",  # Dark Violet
            "#FFB347",  # Papaya Orange
            "#00FF00",  # Lime Green
            "#FF1493",  # Deep Pink
            "#00FFFF",  # Cyan,
			"#FF4500",  # Orange Red
            "#a13830",
			"#e0cd4d",
			"#101010",  # Rich Black
			"#40E0D0",  # Turquoise,
			"#00FA9A",  # Medium Spring Green
            "#6b6a72",
            "#0000FF",  # Pure Blue
            "#7c503a",
			"#7c4e75",
			"#228B22",  # Forest Green
            "#9B870C",  # Dark Yellow
        ]

		# Create a figure with subplots
		fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

		# Extract candidate points from the dataset
		X_candidates = X[candidates]

		for axis in ['top', 'bottom', 'left', 'right']:
			ax1.spines[axis].set_linewidth(2.0)

		# Get unique labels from kmeans_labels
		unique_labels = np.unique(kmeans_labels)

		# Plot each cluster with a different color
		for label in unique_labels:
			# Get the data points for the current cluster
			cluster_points = X[kmeans_labels == label]
			# Determine the color for this cluster
			color = color_list[label % len(color_list)]  # Cycle through color_list if needed
			# Scatter plot for the cluster points
			ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', 
						color=color, edgecolor='none', s=50, alpha=0.7)
		
		X_candidates_other = np.delete(X_candidates, winner_index, axis=0)
		
		#ax1.scatter(X_candidates_other[:,0], X_candidates_other[:,1], c='lime', marker='P', s=400, edgecolors='black')
		#ax1.scatter(X_candidates[winner_index,0], X_candidates[winner_index,1], c='red', marker='*', s=500, edgecolors='black')
		
		# Legend
		leg_data = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Datapoint', markeredgecolor='black', markersize=15)
		leg_winner_candidate = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', label='Winner Candidate', markeredgecolor='black', markersize=15)
		leg_candidates = Line2D([0], [0], marker='P', color='w', markerfacecolor='lime', label='Selected Candidates', markeredgecolor='black', markersize=15)
		#legend = ax1.legend(handles=[leg_data, leg_winner_candidate, leg_candidates], loc='upper center', framealpha=0.8, fontsize=15)

		#legend.get_frame().set_linewidth(2.0)
		#legend.get_frame().set_edgecolor('black')

		# Remove axis ticks for both subplots
		#for ax in (ax2):
		ax1.set_xticks([])
		ax1.set_yticks([])

		# Adjust layout and save the plot
		plt.tight_layout()
		#plt.savefig(f"probabilities_and_clustering_{K - 1}_{self.sampling}.png")
		plt.savefig(f"g{K - 1}_pp_{self.sampling}.png")
		#if(K-1 == 8):
		#plt.show()
		plt.close()

	def plot_solution_2(self, X, kmeans_labels, candidates, K, winner_index):
		color_list = [
			"#008897", "#9400D3", "#FFB347", "#00FF00", "#FF1493", "#00FFFF", "#FF4500",
			"#a13830", "#e0cd4d", "#101010", "#40E0D0", "#00FA9A", "#6b6a72", "#0000FF",
			"#7c503a", "#7c4e75", "#228B22", "#9B870C"
		]

		# First Figure: Candidate Selection Probabilities
		fig1, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 8))
		
		# Extract candidate points
		X_candidates = X[candidates]

		# Plot probabilities
		for axis in ['top', 'bottom', 'left', 'right']:
			ax1.spines[axis].set_linewidth(2.0)

		ax3.axis('off')

		probabilities = self.selection_prob
		norm = Normalize(vmin=np.min(probabilities), vmax=np.max(probabilities))
		cmap = cm.get_cmap('coolwarm')
		point_colors = cmap(norm(probabilities))

		ax1.scatter(X[:, 0], X[:, 1], edgecolors='none', c=point_colors, s=50)
		
		ax1.set_xticks([])
		ax1.set_yticks([])

		sm = plt.cm.ScalarMappable(cmap=cmap)
		sm.set_clim(vmin=np.min(probabilities), vmax=np.max(probabilities))
		cbar = plt.colorbar(sm, ax=ax3, label='Candidate Selection Probability', location="left")
		cbar.ax.tick_params(labelsize=0, length=0)
		cbar.set_label(label='Candidate Selection Probability', fontsize=20)
		
		# Save first figure
		plt.tight_layout()
		plt.savefig(f"probabilities_{K - 1}_{self.sampling}.png")
		plt.close()

		# Second Figure: Clustering Visualization
		fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

		for axis in ['top', 'bottom', 'left', 'right']:
			ax2.spines[axis].set_linewidth(2.0)


		unique_labels = np.unique(kmeans_labels)
		for label in unique_labels:
			cluster_points = X[kmeans_labels == label]
			color = color_list[label % len(color_list)]
			ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}',
						color=color, edgecolor='none', s=50, alpha=0.7)

		ax2.set_xticks([])
		ax2.set_yticks([])

		# Save second figure
		plt.tight_layout()
		plt.savefig(f"clustering_{K - 1}_{self.sampling}.png")
		plt.close()