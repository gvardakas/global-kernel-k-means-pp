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

	def create_adj_matrix(self):
		self.adj_matrix = nx.to_numpy_array(self.G)

	def create_kernel_matrix_from_adj_matrix(self, b=0.1, D=None):
		if(D is None):
			return self.modification.modify_kernel_matrix_ra(self.adj_matrix, b)
		else:
			return self.modification.modify_kernel_matrix_nc(self.adj_matrix, D, b)
	
	def plot_affinity_matrix(self):
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
		nx.draw(self.G, pos, with_labels=False, node_color=node_colors, node_size=5, font_size=12, font_weight='bold')
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

		#self.plot()


	def create_G_from_file(self, file_path):
		edges = []
		with open(file_path, 'r') as f:
			for line in f:
				node1, node2 = map(int, line.split())  # Convert the pair of nodes to integers
				#if node1 != node2:  # Skip self-loop edges (e.g., 0 0, 1 1)
				edges.append((node1, node2))  # Add edge tuple to the list
		self.G = nx.Graph()
		self.G.add_edges_from(edges)

	def create_sample_weights_and_degree_matrix(self):

		# Sum the rows of the adjacency matrix to get the degrees of each node
		self.sample_weights = np.sum(self.adj_matrix, axis=1)

		# Create a diagonal matrix with the degrees
		self.degree_matrix = np.diag(self.sample_weights)	