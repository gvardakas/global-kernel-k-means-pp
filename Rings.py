import numpy as np
from sklearn.datasets import make_circles

class Rings:

	def __init__(self, seed=42):
		self.seed = seed

	def remove_samples(self, N, y, label):
		label_indices = np.where(np.array(y) == label)[0]

		indices_to_remove = np.random.choice(label_indices, size=N, replace=False)

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

		return X, y

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
		