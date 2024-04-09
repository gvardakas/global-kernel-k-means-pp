import numpy as np
import matplotlib.pyplot as plt
import math
from InitialPartition import InitialPartition

class KernelKMeansPP:

    def __init__(self):
        self.initialPartition = InitialPartition()

    def find_cluster_indices(self, array, value):
        return np.where(np.array(array) == value)[0]

    def custom_kernel_kmeans(self, X, centers_indices, initial_partition, kernel_matrix, max_iter=300, tol=1e-4):
        n_samples = X.shape[0]
        n_clusters = len(centers_indices)
        distances = np.zeros((n_clusters,n_samples))
        
        for iter in range(max_iter):
            
            for i in range(n_clusters):
                cluster_indices = self.find_cluster_indices(initial_partition, centers_indices[i])
                n_cluster_samples =  len(cluster_indices)
                stable_sum = (np.sum(kernel_matrix[k, l] for k in cluster_indices for l in cluster_indices)) / pow(n_cluster_samples, 2)
                
                for j in range(n_samples):
                    sample_sum = np.sum(kernel_matrix[j,index] for index in cluster_indices) / n_cluster_samples
                    distances[i,j] = kernel_matrix[j,j] - (2 * sample_sum) + stable_sum 
            
            min_distances = np.min(distances, axis=0)
            total_error = np.sum(min_distances)
            print(f"\n Total Error is: {total_error}!")
            
            next_partition = np.argmin(distances, axis=0)
            are_equal = np.array_equal(initial_partition, next_partition)
            if(are_equal):
                print(f"\n Finished in {iter} iterations!")
                return np.unique(next_partition), next_partition, total_error
            else:
                initial_partition = next_partition
                centers_indices = np.unique(next_partition)

    # 6 are good
    def kernel_kmeans_pp(self, X, K, kernel_matrix, n_init=1, method = 'KkMeans++'):
        min_total_error = math.inf

        for _ in range(n_init):
            centers_indices, partition = self.initialPartition.calculate_initial_partition(K, X, kernel_matrix, method)
            centers_indices, partition, total_error = self.custom_kernel_kmeans(X, centers_indices, initial_partition = partition, kernel_matrix=kernel_matrix, max_iter=300, tol=1e-4)
            print(f"\n centers_indices are: {centers_indices}!")
            if(total_error < min_total_error):
                min_total_error = total_error
                
                plt.scatter(X[:, 0], X[:, 1], c = partition)
                plt.show()
