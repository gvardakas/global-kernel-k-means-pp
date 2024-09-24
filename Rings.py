import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Rings:            
	def __init__(self, colors, seed=42):
		self.colors = colors
		self.seed = seed
		np.random.seed(self.seed)

	def make_multiple_rings_with_gaussians(self, centers_coordinates, n_samples=100, radius=10, noise=0.05, gaussian_samples=50):
		X_list, y_list = [], []
		label_offset = 0
        
		for center_coordinates in centers_coordinates:
            # Generate ring
			X_ring = self.generate_circle(n_samples, radius, noise)
			y_ring = np.full(X_ring.shape[0], label_offset)  # Label for the ring
			X_ring = self.move_rings(center_coordinates, X_ring)
			X_list.append(X_ring)
			y_list.append(y_ring)
            
            # Generate 2 Gaussians inside the ring
			gaussians_X, gaussians_y = self.generate_gaussians(
                gaussian_samples,
                means=[(center_coordinates[0] - 1, center_coordinates[1]), 
                       (center_coordinates[0] + 1, center_coordinates[1])],  # Centered near the ring's center
                cov=[[0.1, 0], [0, 0.1]],  # Covariance matrix
                labels=[label_offset + 1, label_offset + 2]  # Unique labels for Gaussians
            )

			X_list.append(gaussians_X)
			y_list.append(gaussians_y)

			label_offset = label_offset + 3
        
		# Concatenate all parts
		X = np.concatenate(X_list)
		y = np.concatenate(y_list)

		self.plot(X, y)
		return X, y
	
	def plot(self, X, labels_):
		plt.scatter(X[:, 0], X[:, 1], c=labels_, cmap=ListedColormap(self.colors))
		plt.show()

	def remove_samples_with_specific_label(self, X, y, label):
		indices_to_remove = np.where(np.array(y) == label)[0]
		X = np.delete(X, indices_to_remove, axis=0)
		y = np.delete(y, indices_to_remove)

		return X,y
	
	def move_rings(self, center_coordinates, X):
		X[:, 0] += center_coordinates[0]  
		X[:, 1] += center_coordinates[1] 

		return X
	
	def adjust_labels(self, y, id):
		return np.where(y == 0, id, id + 1)

	def concatenate_pairs(self, pairs):
		X = np.concatenate([X for X, _ in pairs])
		y = np.concatenate([y for _, y in pairs])

		self.plot(X, y)

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
				y = np.full(X.shape[0], i)  # Assign unique label for each ring
				X_list.append(X)
				y_list.append(y)
            
			# Concatenate all rings
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
			
			pairs.append((X, y))
			label += 2

		return self.concatenate_pairs(pairs)	
		