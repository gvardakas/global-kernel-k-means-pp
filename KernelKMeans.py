import numpy as np
import math
from Initialization import Initialization
from Common_Modules.Evaluation import Evaluator

class KernelKMeans:

    def __init__(self):
        self.initialization = Initialization()
        self.evaluator = Evaluator()

    def find_cluster_indices(self, labels_, cluster_label):
        return np.where(labels_ == cluster_label)[0]

    def custom_kernel_kmeans(self, N, clusters_labels, initial_labels_, kernel_matrix):
        n_clusters = len(clusters_labels)
        distances = np.zeros((n_clusters, N))
        iter = 0
        previous_labels_ = initial_labels_
            
        while(True):

            for i in range(n_clusters):
                cluster_indices = self.find_cluster_indices(previous_labels_, clusters_labels[i])
                n_cluster_samples =  len(cluster_indices)

                stable_sum = (np.sum(kernel_matrix[k, l] for k in cluster_indices for l in cluster_indices)) / pow(n_cluster_samples, 2)

                for j in range(N):
                    sample_sum = np.sum(kernel_matrix[j,index] for index in cluster_indices) / n_cluster_samples

                    distances[i,j] = kernel_matrix[j,j] - (2 * sample_sum) + stable_sum 
            
            min_distances = np.min(distances, axis=0)
            total_error = np.sum(min_distances)
            
            current_labels_ = np.argmin(distances, axis=0)
            are_equal = np.array_equal(previous_labels_, current_labels_)
            
            ### OK
            if(are_equal):
                print(f'Finished in Iter: {iter} Cl L: {total_error:.4f}')
                return current_labels_, total_error

            print(f'Iter: {iter} Cl L: {total_error:.4f}')
            previous_labels_ = current_labels_
            iter += 1 

    def custom_kernel_kmeans_qq(self, N, clusters_labels, initial_labels_, kernel_matrix):
        n_clusters = len(clusters_labels)
        distances = np.zeros((n_clusters, N))
        iter = 0
        previous_labels_ = initial_labels_
            
        while(True):

            # Precompute kernel_matrix[j,j] and store it in a variable
            kernel_diag = np.diag(kernel_matrix)

            iter = 0
            while True:
                distances = np.zeros((n_clusters, N))
                for i in range(n_clusters):
                    cluster_indices = self.find_cluster_indices(previous_labels_, clusters_labels[i])
                    n_cluster_samples = len(cluster_indices)

                    stable_sum = np.sum(kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)

                    sample_sums = np.sum(kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

                    distances[i] = kernel_diag - 2 * sample_sums + stable_sum

                min_distances = np.min(distances, axis=0)
                total_error = np.sum(min_distances)

                current_labels_ = np.argmin(distances, axis=0)
                are_equal = np.array_equal(previous_labels_, current_labels_)

                if are_equal:
                    print(f'Finished in Iter: {iter} Cl L: {total_error:.4f}')
                    return current_labels_, total_error

                print(f'Iter: {iter} Cl L: {total_error:.4f}')
                previous_labels_ = current_labels_
                iter += 1

        

    def kernel_kmeans(self, X, K, kernel_matrix, n_init=10, method = 'k-means++'):
        min_total_error = math.inf
        best_labels_ = []
        N = X.shape[0]
        
        for _ in range(n_init):
            clusters_labels, initial_labels_ = self.initialization.calculate_initial_partition(K, X, kernel_matrix, method)
            labels_, total_error = self.custom_kernel_kmeans(N, clusters_labels, initial_labels_ = initial_labels_, kernel_matrix=kernel_matrix)
            
            if(total_error < min_total_error):
                min_total_error = total_error
                best_labels_ = labels_

        return min_total_error, best_labels_        
