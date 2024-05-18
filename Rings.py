import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Rings:

	def __init__(self, colors, seed=42):
		self.colors = colors
		self.seed = seed

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
		