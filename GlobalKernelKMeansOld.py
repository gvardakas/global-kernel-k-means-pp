import math
from Initialization import Initialization
from KernelKMeans import KernelKMeans

class GlobalKernelKMeansOld:

    def __init__(self):
        self.initialization = Initialization()
        self.kernelKMeans =  KernelKMeans()

    def global_kernel_kmeans(self, K, X, kernel_matrix, method = 'global'):
        min_total_error = math.inf
        best_labels_ = []
        best_k_labels_ = []
        total_min_total_errors = []
        total_best_k_labels_ = []
        N = X.shape[0]
        
        for k in range(2, K+1):
            print(f"Now Examining K = {k}")
            min_k_total_error = math.inf
            
            for index in range(N):
                k_total_error, k_labels_ = self.kernelKMeans.kernel_kmeans(X, k, kernel_matrix, 1, method, initial_labels_=best_labels_, index=index)
                
                if(k_total_error < min_k_total_error):
                    min_k_total_error = k_total_error
                    best_k_labels_ = k_labels_
            
            total_min_total_errors.append(min_k_total_error)
            total_best_k_labels_.append(best_k_labels_)
            min_total_error = min_k_total_error
            best_labels_ = best_k_labels_
            
        return min_total_error, best_labels_, total_min_total_errors, total_best_k_labels_     
