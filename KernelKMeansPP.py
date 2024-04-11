import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
from InitialPartition import InitialPartition
from Common_Modules.Evaluation import Evaluator

class KernelKMeansPP:

    def __init__(self):
        self.initialPartition = InitialPartition()
        self.evaluator = Evaluator()

    def find_cluster_indices(self, array, value):
        return np.where(np.array(array) == value)[0]

    def custom_kernel_kmeans(self, X, y, clusters_labels, initial_partition, kernel_matrix, max_iter=300, tol=1e-4):
        n_samples = X.shape[0]
        n_clusters = len(clusters_labels)
        distances = np.zeros((n_clusters,n_samples))
        
        for iter in range(max_iter):
            
            for i in range(n_clusters):
                cluster_indices = self.find_cluster_indices(initial_partition, clusters_labels[i])
                n_cluster_samples =  len(cluster_indices)

                if(n_cluster_samples != 0):
                    stable_sum = (np.sum(kernel_matrix[k, l] for k in cluster_indices for l in cluster_indices)) / pow(n_cluster_samples, 2)
                else:
                    stable_sum = 0

                for j in range(n_samples):

                    if(n_cluster_samples != 0):
                        sample_sum = np.sum(kernel_matrix[j,index] for index in cluster_indices) / n_cluster_samples
                    else:
                        sample_sum = 0
                    
                    distances[i,j] = kernel_matrix[j,j] - (2 * sample_sum) + stable_sum 
            
            min_distances = np.min(distances, axis=0)
            total_error = np.sum(min_distances)
            
            next_partition = np.argmin(distances, axis=0)
            are_equal = np.array_equal(initial_partition, next_partition)
            if(are_equal):
                acc, pur, nmi, ari = self.evaluator.evaluate_model(y, next_partition)
                print(f'Finished in Iter: {iter} Cl L: {total_error:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')
                return next_partition, total_error
            else:
                acc, pur, nmi, ari = self.evaluator.evaluate_model(y, next_partition)
                print(f'Iter: {iter} Cl L: {total_error:.4f} ACC: {acc:.2f} PUR: {pur:.2f} NMI: {nmi:.2f} ARI: {ari:.2f}')

                initial_partition = next_partition

    # 6 are good
    def kernel_kmeans_pp(self, X, y, K, kernel_matrix, n_init=1, method = 'KkMeans++'):
        min_total_error = math.inf
        best_partition = []
        
        for _ in range(n_init):
            clusters_labels, partition = self.initialPartition.calculate_initial_partition(K, X, kernel_matrix, method)
            partition, total_error = self.custom_kernel_kmeans(X, y, clusters_labels, initial_partition = partition, kernel_matrix=kernel_matrix, max_iter=300, tol=1e-4)
            
            if(total_error < min_total_error):
                min_total_error = total_error
                best_partition = partition

        return min_total_error, partition        
