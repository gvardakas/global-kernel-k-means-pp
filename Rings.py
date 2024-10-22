
import scipy.io
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Rings:            
	def __init__(self, seed=42):
		self.seed = seed
		np.random.seed(self.seed)
	
	def make_spiral(self, n_samples=1000, noise=0.05, n_turns=2, separation=2):
		"""Generate a 2D dataset with two interlocking spirals."""
		theta = np.sqrt(np.random.rand(n_samples)) * n_turns * 2 * np.pi  # Random angles
		r = theta + noise * np.random.randn(n_samples)  # Radii with noise

		# First spiral (positive labels)
		X1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
		y1 = np.zeros(n_samples, dtype=int)

		# Second spiral (negative labels) - rotate by 180 degrees
		X2 = np.c_[r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)]
		y2 = np.ones(n_samples, dtype=int)

		# Concatenate spirals and labels
		X = np.vstack((X1, X2))
		y = np.hstack((y1, y2))

		return X, y

	def make_multiple_rings_with_gaussians(self, centers_coordinates, n_samples=100, radius=10, noise=0.05, gaussian_samples=50):
		X_list, y_list = [], []
		label_offset = 0
        
		for center_coordinates in centers_coordinates:
			X_ring = self.generate_circle(n_samples, radius, noise)
			y_ring = np.full(X_ring.shape[0], label_offset)
			X_ring = self.move_rings(center_coordinates, X_ring)
			X_list.append(X_ring)
			y_list.append(y_ring)
            
			gaussians_X, gaussians_y = self.generate_gaussians(
                gaussian_samples,
                means=[(center_coordinates[0] - 1, center_coordinates[1]), 
                       (center_coordinates[0] + 1, center_coordinates[1])],
                cov=[[0.1, 0], [0, 0.1]],
                labels=[label_offset + 1, label_offset + 2]
            )

			X_list.append(gaussians_X)
			y_list.append(gaussians_y)

			label_offset = label_offset + 3
        
		X = np.concatenate(X_list)
		y = np.concatenate(y_list)

		return X, y
	
	def remove_samples_with_specific_label(self, X, y, label, num_to_remove):
		indices_to_remove = np.where(np.array(y) == label)[0]

		# Determine how many indices to actually remove
		num_available = len(indices_to_remove)
		num_removed = min(num_to_remove, num_available)  # Remove only available samples

		# Select indices to remove
		indices_to_remove = indices_to_remove[:num_removed]  # Limit to num_removed
		X = np.delete(X, indices_to_remove, axis=0)
		y = np.delete(y, indices_to_remove)

		return X, y
	
	def move_rings(self, center_coordinates, X):
		X[:, 0] += center_coordinates[0]  
		X[:, 1] += center_coordinates[1] 

		return X
	
	def adjust_labels(self, y, id):
		return np.where(y == 0, id, id + 1)

	def concatenate_pairs(self, pairs):
		X = np.concatenate([X for X, _ in pairs])
		y = np.concatenate([y for _, y in pairs])

		return X, y

	def generate_circle(self, n_samples, radius, noise):
		angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
		X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
		X += noise * np.random.randn(n_samples, 2)
		return X
    
	def generate_gaussians(self, n_samples, means, cov, labels):
		X = np.vstack([np.random.multivariate_normal(mean, cov, n_samples) for mean in means])
		y = np.hstack([[label]*n_samples for label in labels])
		return X, y
	
	def make_concentric_rings(self, centers_coordinates, n_samples=300, radii=[1, 2, 3], noise=0.05):
		pairs = []
		label = 0
		for center_coordinates in centers_coordinates:
			X_list, y_list = [], []
			
			for i, radius in enumerate(radii):
				X = self.generate_circle(n_samples // len(radii), radius, noise)
				y = np.full(X.shape[0], i)
				X_list.append(X)
				y_list.append(y)
            
			X = np.concatenate(X_list)
			y = np.concatenate(y_list)
			X = self.move_rings(center_coordinates, X)
			
			pairs.append((X, y))
			label += len(radii)
		
		return self.concatenate_pairs(pairs)
	
	def make_rings_pairs(self, centers_coordinates, n_samples=100, factor=0.2, noise=0.05):
		pairs = []
		label = 0
		
		for _, center_coordinates in enumerate(centers_coordinates):
			X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=self.seed)
			
			X = self.move_rings(center_coordinates, X)
			y = self.adjust_labels(y, label)
			
			#X, y = self.remove_samples_with_specific_label(X, y, label-1, round(n_samples/4))
			
			pairs.append((X, y))
			label += 2

		return self.concatenate_pairs(pairs)	
	
	def global_kernel_k_means_three_rings(self):
		X = np.array(scipy.io.loadmat('3circles_dataset.mat')['Dataset'])
		y = np.loadtxt('array.txt').astype(int)
		kernel_matrix = np.array(scipy.io.loadmat('3circles_kernel_matrix.mat')['K'])

		return X, y, kernel_matrix
		