import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Common_Modules.General_Functions import General_Functions

class Visualization:

    def __init__(self):
        self.color_list = self.init_color_list()

    def init_color_list(self):
        color_list = [
            "#a13830",
            "#008897",
            "#FF4500",  # Orange Red
            "#6b6a72",
            "#FFB347",  # Papaya Orange
            "#0000FF",  # Pure Blue
            "#7c503a",
            "#228B22",  # Forest Green
            "#9B870C",  # Dark Yellow
            "#9400D3",  # Dark Violet
            "#101010",  # Rich Black
            "#00FF00",  # Lime Green
            "#FF1493",  # Deep Pink
            "#00FFFF",  # Cyan,
            "#7c4e75",
            "#00FA9A",  # Medium Spring Green
            "#e0cd4d",
            "#40E0D0",  # Turquoise,
        ]

        return color_list
    
    def plot_image(self, image, label):
        plt.imshow(image.squeeze(), cmap='gray')  
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    def plot_collage(self, images, num_x_images, num_y_images, image_size, data_dir_path):
    
        _, axes = plt.subplots(num_x_images, num_y_images, figsize=(image_size[0], image_size[1]), facecolor = 'black')

        for i in range(num_x_images):
            for j in range(num_y_images):
                index = i * num_y_images + j
                image = images[index].reshape(image_size[0], image_size[1])
                
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(data_dir_path + "/Experiments/Collage.png", facecolor = 'white')
        plt.show()

    
    def plot_tsne(self, data, y_true, cluster_centers=None, data_dir_path=None):
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        if(cluster_centers == None):
          cluster_centers = np.zeros(data.shape)

        tsne_embeddings = tsne.fit_transform(np.concatenate((cluster_centers, data)))

        n_clusters = cluster_centers.shape[0]
        unique_labels = np.unique(y_true).astype(int)

        plt.figure(figsize=(10, 10))
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = tsne_embeddings[n_clusters:][selected_indexes, 0]
            y = tsne_embeddings[n_clusters:][selected_indexes, 1]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            plt.scatter(x=x, y=y, c=c) #, edgecolors='black')

        # Plot cluster centers
        if(cluster_centers.all() != 0):
            plt.scatter(tsne_embeddings[:n_clusters, 0], tsne_embeddings[:n_clusters, 1], c='red', marker='x', s=500, linewidths=3, label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks
        
        plt.axis('off')
        
        plt.tight_layout()
        
        plt.savefig(data_dir_path + "_TSNE.png")
        plt.show() 
    
    def plot(self, data, y_true, cluster_centers=None, data_dir_path=None):
        if(cluster_centers == None):
          cluster_centers = np.zeros(data.shape[0])

        n_clusters = cluster_centers.shape[0]
        unique_labels = np.unique(y_true).astype(int)

        plt.figure(figsize=(10, 10))
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = data[selected_indexes, 0]
            y = data[selected_indexes, 1]
            c = self.color_list[label_id]
            plt.scatter(x=x, y=y, c=[c] * selected_indexes.shape[0]) #, edgecolors='silver')

        # Plot cluster centers
        if(cluster_centers.all() != 0):
            plt.scatter(cluster_centers[:n_clusters, 0], cluster_centers[:n_clusters, 1], c='red', marker='x', s=500, linewidths=3, label='Cluster Centers')

        # Remove x-axis numbering and label
        plt.xticks([])  # Pass an empty list to remove ticks

        # Remove y-axis numbering and label
        plt.yticks([])  # Pass an empty list to remove ticks

        plt.tight_layout()
        
        plt.axis('off')
        
        plt.savefig(data_dir_path + ".png")
        plt.show()   

    def plot_3D(self, data, y_true, y_predict, cluster_centers, data_dir_path):
        unique_labels = np.unique(y_true).astype(int)
        
        # Cluster with TSNE
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for label_id in unique_labels:
            selected_indexes = np.where(y_true == label_id)[0]
            x = data[selected_indexes, 0]
            y = data[selected_indexes, 1]
            z = data[selected_indexes, 2]
            c = [self.color_list[label_id]] * selected_indexes.shape[0]
            ax.scatter(x, y, z, c=c)
        
        # Set labels
        ax.set_xlabel('$x$', fontsize = 10)
        ax.set_ylabel('$y$', fontsize = 10)
        ax.set_zlabel('$z$', fontsize = 10)
        
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        
        plt.tight_layout()
        
        exp_dir_path, experiment_name = General_Functions().save_plot(data_dir_path, "3D")
        plt.savefig(exp_dir_path + "/" + experiment_name + "_3D.png")
        plt.show()

