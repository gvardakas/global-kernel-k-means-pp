import numpy as np
import math
from Initialization import Initialization

class KernelKMeans:

    def __init__(self, K, kernel_matrix, n_init=10, method='k-means++', initial_labels_=None):
        self.initialization = Initialization()
        self.K = K
        self.kernel_matrix = kernel_matrix
        self.n_init = n_init
        self.method = method
        self.initial_labels_ = initial_labels_
    

    def find_cluster_indices(self, labels_, cluster_label):
        return np.where(labels_ == cluster_label)[0]

    def kernel_kmeans_functionallity(self, N, initial_labels_, kernel_matrix):
        clusters_identities, initial_labels_ = self.initialization.scale_partition(self.K, initial_labels_)
        distances = np.zeros((self.K, N))
        iter = 0
        previous_labels_ = initial_labels_
            
        while(True):

            # Precompute kernel_matrix[j,j] and store it in a variable
            kernel_diag = np.diag(kernel_matrix)

            iter = 0
            while True:
                distances = np.zeros((self.K, N))
                for i in range(self.K):
                    cluster_indices = self.find_cluster_indices(previous_labels_, clusters_identities[i])

                    n_cluster_samples = len(cluster_indices)
                    if(n_cluster_samples ==0):
                        stable_sum = 0
                        sample_sums = 0
                    else:
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

    def fit(self, X):
        self.min_total_error = math.inf
        self.best_labels_ = []
        N = X.shape[0]
        
        for _ in range(self.n_init):
            if(self.initial_labels_ is None):    
                self.initial_labels_ = self.initialization.calculate_initial_partition(self.K, X, self.kernel_matrix, self.method)

            labels_, total_error = self.kernel_kmeans_functionallity(N, initial_labels_ = self.initial_labels_, kernel_matrix=self.kernel_matrix)
            
            self.initial_labels_ = None

            if(total_error < self.min_total_error):
                self.min_total_error = total_error
                self.best_labels_ = labels_