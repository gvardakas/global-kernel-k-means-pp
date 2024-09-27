from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import math
import time
from Initialization import Initialization

class _BaseKernelKMeans(BaseEstimator, ClusterMixin, TransformerMixin, ABC):
    """Base class for Kernel K-Means, Kernel K-Means++ and future (or past) variants.

        Parameters
        ----------
            n_clusters (int) : The number of clusters to form and the number of centroids to generate.
            kernel_matrix (numpy.array) : The kernel matrix array.
            n_init (int) : The number of initializations to perform.
            init (string) : The initialization method to use.
            initial_labels_ (numpy.array) : The initial labels array.
            tol (float) : The convergence tolerance to use.
            verbose (int) : The verbosity mode. 
            N (int) : The number of points.

        Attributes
        ----------
            n_iter_ (int) : The total number of Kernel kMeans iterations.
            labels_ (numpy.array) : The array to store cluster labels.
            inertia_ (float) : The variable to store inertia value.
    """
    def __init__(self, n_clusters, kernel_matrix, n_init, init, initial_labels_, tol, verbose):
        self.n_clusters = n_clusters
        self.kernel_matrix = kernel_matrix
        self.n_init = n_init
        self.init = init
        self.initial_labels_ = initial_labels_
        self.tol = tol
        self.verbose = verbose
        self.N = kernel_matrix.shape[0]

        self.initialization = Initialization()
        self.n_iter_ = 0
        self.labels_ = []
        self.inertia_ = math.inf
        self.execution_times_ = {}
        self.n_iters_ = {}

    @abstractmethod	
    def fit(self, X=None, y=None, sample_weight=None):
        """Abstract method for fitting the model to the data.

        Parameters
        ----------
            X (array-like) : Ignored.
            y : Ignored
            sample_weight : Ignored

        Returns
        ----------
            self: Fitted estimator.
        """
        return self

    def predict(self, X=None):
        """Predict cluster labels for data X with n_clusters clusters.

        Parameters
        ----------
            X : Ignored. Models should already be fitted.

        Returns
        ----------
            labels_ (array) : Cluster labels.
        """
        return self.labels_

    def fit_predict(self, X=None, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).


        Parameters
        ----------
            X (array-like) : Ignored.

        Returns
        ----------
            labels_ (array) : Cluster labels.
        """
        self.fit(X, y, sample_weight)
        return self.predict(X)

class KernelKMeans(_BaseKernelKMeans):

    def __init__(self, n_clusters, kernel_matrix, n_init=10, init='k-means++', initial_labels_=None, tol=1e-20, verbose=0):
        super().__init__(
            n_clusters=n_clusters,
            kernel_matrix=kernel_matrix,
            n_init=n_init,
            init=init,
            initial_labels_=initial_labels_,
            verbose=verbose,
            tol = tol
        )
    
    def calculate_ground_truth_error(self, labels_):
        kernel_diag = np.diag(self.kernel_matrix)
        distances = np.zeros((self.n_clusters, self.N))
        
        clusters_identities, labels_ = self.initialization.scale_partition(self.n_clusters, np.copy(labels_))
                
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels_ == clusters_identities[i])[0]

            n_cluster_samples = len(cluster_indices)
            stable_sum = np.sum(self.kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
            sample_sums = np.sum(self.kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples

            distances[i] = kernel_diag - 2 * sample_sums + stable_sum

        min_distances = np.min(distances, axis=0)
        
        return np.sum(min_distances)
    
    def __kernel_kmeans_functionallity(self, initial_labels_, kernel_matrix):
        self.n_iter_ = 0
        kernel_diag = np.diag(kernel_matrix)
        distances = np.zeros((self.n_clusters, self.N))

        previous_inertia_ = math.inf
        clusters_identities, previous_labels_ = self.initialization.scale_partition(self.n_clusters, np.copy(initial_labels_))
        
        while True:
            for i in range(self.n_clusters):
                cluster_indices = np.where(previous_labels_ == clusters_identities[i])[0]

                n_cluster_samples = len(cluster_indices)
                
                if(n_cluster_samples == 0):
                    return None, None
                    
                stable_sum = np.sum(kernel_matrix[np.ix_(cluster_indices, cluster_indices)]) / (n_cluster_samples ** 2)
                sample_sums = np.sum(kernel_matrix[:, cluster_indices], axis=1) / n_cluster_samples
                
                distances[i] = kernel_diag - 2 * sample_sums + stable_sum

            self.min_distances = np.min(distances, axis=0)
            current_inertia_ = np.sum(self.min_distances)
            current_labels_ = np.argmin(distances, axis=0)
            
            if (abs(current_inertia_ - previous_inertia_) < self.tol):   
                #if(self.verbose > 1):
                    #print(f'Finished in Iter: {self.n_iter_} MSE: {current_inertia_:.4f}')

                return current_labels_, current_inertia_

            if(self.verbose > 2):
                print(f'Iter: {self.n_iter_} MSE: {current_inertia_:.4f}')
            
            previous_inertia_ = current_inertia_
            previous_labels_ = np.copy(current_labels_)

            self.n_iter_ += 1

    def fit(self):
        self.inertia_ = math.inf
        self.labels_ = []
        i = 0
        total_inertia_ = 0
        while (i < self.n_init):
            if(self.verbose > 1):
                print(f'Execution {i} of Kernel k-Means')
            
            start_time = time.time()	
            
            if(self.initial_labels_ is None):
                self.initial_labels_ = self.initialization.calculate_initial_partition(self.n_clusters, self.N, self.kernel_matrix, self.init)
            
            current_labels_, current_inertia_ = self.__kernel_kmeans_functionallity(initial_labels_ = np.copy(self.initial_labels_), kernel_matrix=self.kernel_matrix)
             
            self.initial_labels_ = None
            
            if(current_inertia_ == None and current_labels_ == None):
                continue 

            if(current_inertia_ < self.inertia_):
                self.inertia_ = current_inertia_
                self.labels_ = np.copy(current_labels_)
            total_inertia_ += current_inertia_
            self.execution_times_[i] = time.time() - start_time
            self.n_iters_[i] = self.n_iter_

            if(self.verbose > 1):
                print(f'Execution {i} of Kernel k-Means MSE: {current_inertia_} in {self.execution_times_[i]}s')        
            i+=1
        if self.verbose > 0: 
            print(f'Total execution time was {sum(self.execution_times_.values())}s')
        #print(f"Avg Inertia was {total_inertia_/self.n_init}")
        return self        