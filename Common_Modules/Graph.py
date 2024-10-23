import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Common_Modules.Modification import Modification 

class Graph:

	def __init__(self):
		self.modification = Modification()
	
	def create_adj_matrix(self):
		self.adj_matrix = nx.to_numpy_array(self.G)

	def create_kernel_matrix_from_adj_matrix(self, b=0.1, D=None):
		if(D is None):
			return self.modification.modify_kernel_matrix_ra(self.adj_matrix, b)
		else:
			return self.modification.modify_kernel_matrix_nc(self.adj_matrix, D, b)
	
	def create_G_from_file(self, file_path):
		edges = []
		with open(file_path, 'r') as f:
			for line in f:
				node1, node2 = map(int, line.split())
				edges.append((node1, node2))
		
		self.G = nx.Graph()
		self.G.add_edges_from(edges)

		self.create_adj_matrix()

		self.create_sample_weights_and_degree_matrix()

	def create_sample_weights_and_degree_matrix(self):
		self.sample_weights = np.sum(self.adj_matrix, axis=1)

		self.degree_matrix = np.diag(self.sample_weights)