
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Common_Modules.Modification import Modification 

class Graph:

	def __init__(self, n_communities, n_nodes_per_community, p_intra, p_inter, colors):
		self.modification = Modification()
		self.n_communities = n_communities
		self.n_nodes_per_community = n_nodes_per_community
		self.p_intra = p_intra
		self.p_inter = p_inter
		self.colors = colors
		self.create()

	def create_kernel_matrix_from_adj_matrix(self, b=0.0001):
		self.adj_matrix = nx.to_numpy_array(self.G)
		print(self.adj_matrix)
		for i in range(self.adj_matrix.shape[0]):
			for j in range (self.adj_matrix.shape[1]):
				if(self.adj_matrix[i][j] != 0):
					self.adj_matrix[i][j] = self.adj_matrix[i][j] + i
					self.adj_matrix[j][i] = self.adj_matrix[i][j]
		
		print(self.adj_matrix)
		if(not self.modification.check_for_positive_defined_matrix(self.adj_matrix)):
			self.kernel_matrix = self.modification.modify_kernel_matrix(self.adj_matrix, 0.0001)
		else:
			self.kernel_matrix = self.adj_matrix
		
		return self.kernel_matrix	
	
	def plot_affinity_matrix(self):
		# Affinity matrix
		plt.figure(figsize=(8, 6))
		plt.imshow(self.adj_matrix, cmap='coolwarm', origin='upper')
		plt.colorbar()
		plt.title('Affinity Matrix')
		plt.xlabel('Node Index')
		plt.ylabel('Node Index')
		plt.show()
	
	def plot(self):
		plt.figure(figsize=(10, 8))
		pos = nx.spring_layout(self.G)
		nx.draw(self.G, pos, with_labels=False, node_color='skyblue', node_size=300, font_size=12, font_weight='bold')
		plt.title(f"Graph with {self.n_communities} Communities")
		plt.show()

	def plot_clusters(self, labels_):
		# Extract node colors based on labels
		node_colors = [self.colors[labels_[node]] for node in self.G.nodes()]

		# Plot the graph
		plt.figure(figsize=(8, 6))
		pos = nx.spring_layout(self.G)
		nx.draw(self.G, pos, with_labels=False, node_color=node_colors, node_size=500, font_size=12, font_weight='bold')
		plt.title("Graph Colored by Labels")
		plt.show()
	
	def create(self):
		# Construct the probability matrix
		p_matrix = np.ones((self.n_communities, self.n_communities)) * self.p_inter
		np.fill_diagonal(p_matrix, self.p_intra)

		# Create a graph with stochastic block model
		self.G = nx.generators.community.stochastic_block_model(
    		sizes=[self.n_nodes_per_community] * self.n_communities,  # Sizes of communities
    		p=p_matrix  # Probability matrix for inter-community edges
		)

		self.plot()