import numpy as np
import math
from Initialization import Initialization

class KernelKMeans:

    def __init__(self, n_clusters, kernel_matrix, n_init=10, init='k-means++', initial_labels_=None):
        self.initialization = Initialization()
        self.n_clusters = n_clusters
        self.kernel_matrix = kernel_matrix
        self.n_init = n_init
        self.init = init
        self.initial_labels_ = initial_labels_
    
    def calculate_ground_truth_error(self, labels_):
        clusters_identities, labels_ = self.initialization.scale_partition(self.n_clusters, labels_)
        N = len(labels_)
        kernel_diag = np.diag(self.kernel_matrix)
        distances = np.zeros((self.n_clusters, N))
        
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels_ == clusters_identities[i])[0]

            n_cluster_samples = len(cluster_indices)
            stable_sum = np.sum(self.kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
            sample_sums = np.sum(self.kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

            distances[i] = kernel_diag - 2 * sample_sums + stable_sum

        min_distances = np.min(distances, axis=0)
        
        return np.sum(min_distances)

    def kernel_kmeans_functionallity(self, N, initial_labels_, kernel_matrix):
        clusters_identities, initial_labels_ = self.initialization.scale_partition(self.n_clusters, initial_labels_)
        distances = np.zeros((self.n_clusters, N))
        previous_labels_ = initial_labels_
        
        kernel_diag = np.diag(kernel_matrix)
        self.n_iter_ = 0
        
        while True:
                distances = np.zeros((self.n_clusters, N))
                for i in range(self.n_clusters):
                    cluster_indices = np.where(previous_labels_ == clusters_identities[i])[0]
                    
                    n_cluster_samples = len(cluster_indices)
                    stable_sum = np.sum(kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
                    sample_sums = np.sum(kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

                    distances[i] = kernel_diag - 2 * sample_sums + stable_sum

                self.min_distances = np.min(distances, axis=0)
                inertia_ = np.sum(self.min_distances)

                current_labels_ = np.argmin(distances, axis=0)
                are_equal = np.array_equal(previous_labels_, current_labels_)

                if are_equal:
                    #print(f'Finished in Iter: {self.n_iter_} Cl L: {inertia_:.4f}')
                    return current_labels_, inertia_

                #print(f'Iter: {self.n_iter_} Cl L: {inertia_:.4f}')
                previous_labels_ = current_labels_
                self.n_iter_ += 1

    def fit(self, X):
        self.inertia_ = math.inf
        self.labels_ = []
        N = X.shape[0]
        
        for _ in range(self.n_init):
            if(self.initial_labels_ is None):    
                self.initial_labels_ = self.initialization.calculate_initial_partition(self.n_clusters, X, self.kernel_matrix, self.init)

            current_labels_, current_inertia_ = self.kernel_kmeans_functionallity(N, initial_labels_ = self.initial_labels_, kernel_matrix=self.kernel_matrix)
            
            self.initial_labels_ = None

            if(current_inertia_ < self.inertia_):
                self.inertia_ = current_inertia_
                self.labels_ = current_labels_
        
        return self        